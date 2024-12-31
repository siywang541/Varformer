import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import dist

from archs.var_vqgan_arch import *
from utils import get_root_logger
from utils.registry import ARCH_REGISTRY
# from .var import VAR
from functools import partial
from archs.var_modules import SharedAdaLin, AdaLNSelfAttn,AdaLNBeforeHead, CrossAttention1, CrossAttentionAR, sample_with_top_k_top_p_
ITen = torch.LongTensor




@ARCH_REGISTRY.register()
class VarCodeFormer2(VarVQAutoEncoder):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer2, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        






        


        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            k = ratio**8
            if k > 0.3:
                k = 0.1
            if k < 0.00015:
                k = 0.00015

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                # next_token_map = next_token_map2*(1-k) + next_token_map*k
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(f_hat), logits_BlV_all, f_hat
        
        elif code_only: # for training stage II
            return logits_BlV_all, f_hat
        
        return self.fhat_to_img(f_hat).add_(1).mul_(0.5), logits_BlV_all, f_hat 
        
    






from archs.var_vqgan_arch import MainDecoder, MainDecoder9

@ARCH_REGISTRY.register()
class VarCodeFormer3(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer3, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        


        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder(**ddconfig)
        


        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(f_hat))
        x = self.main_decoder(dec_res_feats, f_hat)
        return x.clamp_(-1, 1)
    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)

            k = ratio**8
            if k > 0.3:
                k = 0.1
            if k < 0.00015:
                k = 0.00015

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(f_hat), logits_BlV_all, lq_feat
        
        elif code_only: # for training stage II
            return logits_BlV_all, lq_feat
        
        return self.fhat_to_img(f_hat).add_(1).mul_(0.5), logits_BlV_all, lq_feat 
        





@ARCH_REGISTRY.register()
class VarCodeFormer4(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer4, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder(**ddconfig)
        


        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, f_hat: torch.Tensor, prompt_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        x = self.main_decoder(dec_res_feats, f_hat)
        return x.clamp_(-1, 1)
    

    def forward(self, inp_B3HW, prompt_img , code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        prompt_idx = self.img_to_idxBl(prompt_img)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > 2:
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            else:
                idx_Bl = prompt_idx[si]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(lq_feat,f_hat), lq_feat
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(lq_feat,f_hat).add_(1).mul_(0.5), lq_feat 
        






@ARCH_REGISTRY.register()
class VarCodeFormer5(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer5, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder(**ddconfig)
        


        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, f_hat: torch.Tensor, prompt_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        x = self.main_decoder(dec_res_feats, f_hat)
        return x.clamp_(-1, 1)
    

    def forward(self, inp_B3HW, prompt_img , code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_encoder_out = self.img_to_encoder_out(inp_B3HW)
        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)
        # x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        prompt_idx = self.img_to_idxBl(prompt_img)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > 2:
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            else:
                idx_Bl = prompt_idx[si]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(x_encoder_out,f_hat), lq_feat
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out,f_hat).add_(1).mul_(0.5), lq_feat 
        






@ARCH_REGISTRY.register()
class VarCodeFormer6(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer6, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder(**ddconfig)
        


        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, f_hat: torch.Tensor, prompt_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        x = self.main_decoder(dec_res_feats, f_hat)
        return x.clamp_(-1, 1)
    

    def forward(self, inp_B3HW, prompt_img , code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_encoder_out = self.img_to_encoder_out(inp_B3HW)
        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)
        # x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        prompt_idx = self.img_to_idxBl(prompt_img)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > 1:
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            else:
                idx_Bl = prompt_idx[si]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(x_encoder_out,f_hat), lq_feat
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out,f_hat).add_(1).mul_(0.5), lq_feat 
        





@ARCH_REGISTRY.register()
class VarCodeFormer7(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer7, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder(**ddconfig)
        


        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.bf_norm = nn.LayerNorm(embed_dim * 2)
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, f_hat: torch.Tensor, prompt_hat: torch.Tensor):
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        x = self.main_decoder(dec_res_feats, f_hat)
        return x.clamp_(-1, 1)
    

    def forward(self, inp_B3HW, prompt_img , code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        x_encoder_out = self.img_to_encoder_out(inp_B3HW)
        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)
        # x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        prompt_idx = self.img_to_idxBl(prompt_img)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > 1:
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            else:
                idx_Bl = prompt_idx[si]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                out = self.bf_norm(torch.cat([next_token_map2, next_token_map],2))
                next_token_map = self.cat_linear(out)
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(x_encoder_out,f_hat), lq_feat
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out,f_hat).add_(1).mul_(0.5), lq_feat 
        







from archs.var_vqgan_arch import MainDecoder, MainDecoder2
@ARCH_REGISTRY.register()
class VarCodeFormer8(VarVQAutoEncoder2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                
                ):
        super(VarCodeFormer8, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
            
        )
        self.main_decoder = MainDecoder2(**ddconfig)
        



        
        # # encoder res --> decoder

        self.connect_list = connect_list
        
        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        



        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()


    def fhat_to_img(self, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
        

    def forward(self, inp_B3HW, prompt_img , code_only=False, pixel_l = False, cfg=1.5) -> torch.Tensor:  # returns logits_BLV

        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)


        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)
        # x_idx_Bl = self.img_to_idxBl(inp_B3HW)
        prompt_idx = self.img_to_idxBl(prompt_img)
        
        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            if si > 1:
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            else:
                idx_Bl = prompt_idx[si]

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                
                next_token_map = self.cat_linear(torch.cat([next_token_map2, next_token_map],2))
                

                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if pixel_l:
            return self.fhat_to_img(f_hat,lq_feat, enc_feat_dict, fuse_list), lq_feat
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(f_hat,lq_feat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 
        


from archs.RAC import RcaMoudle
@ARCH_REGISTRY.register()
class VarCodeFormer9(VarVQAutoEncoder2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                ):
        super(VarCodeFormer9, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,
            connect_list = connect_list, # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder2(**ddconfig)
        

        # refuse rec and syn prompt
        self.cat_rec_syn_p = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4) #ResBlock(64, 32)
        self.cat_rec_syn_m = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4)
        self.cat_rec_syn_p_ratio = nn.Parameter(torch.ones(10))

        
        # # encoder res --> decoder

        self.connect_list = connect_list
        
        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(depth//3)
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        
        prompt_hat = self.cat_rec_syn_p(f_hat, prompt_hat)
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        f_hat = self.cat_rec_syn_m(f_hat, x_encoder_out)
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
    
    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            r_f_hat.add_(h_up*self.cat_rec_syn_p_ratio[si])

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 
        


from archs.AIEM import EnhanceLayers
@ARCH_REGISTRY.register()
class VarCodeFormer10(VarVQAutoEncoder2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                ):
        super(VarCodeFormer10, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        
        self.n_layers =  n_layers       
        self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)
        


        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,
            connect_list = connect_list, # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder2(**ddconfig)
        

        # refuse rec and syn prompt
        self.cat_rec_syn_p = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4) #ResBlock(64, 32)
        self.cat_rec_syn_m = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4)
        self.cat_rec_syn_p_ratio = nn.Parameter(torch.ones(10))

        
        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(depth//3)
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        
        prompt_hat = self.cat_rec_syn_p(f_hat, prompt_hat)
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        f_hat = self.cat_rec_syn_m(f_hat, x_encoder_out)
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
    
    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)

        x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            r_f_hat.add_(h_up*self.cat_rec_syn_p_ratio[si])

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 
        


@ARCH_REGISTRY.register()
class VarCodeFormer11(VarVQAutoEncoder2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                ):
        super(VarCodeFormer11, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)
        

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,
            connect_list = connect_list, # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder7(**ddconfig)
        

        # refuse rec and syn prompt
        self.cat_rec_syn_p = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4) #ResBlock(64, 32)
        self.cat_rec_syn_m = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4)
        self.cat_rec_syn_p_ratio = nn.Parameter(torch.ones(10))

        
        # # encoder res --> decoder

        self.connect_list = connect_list
        
        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(depth//3)
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        
        prompt_hat = self.cat_rec_syn_p(f_hat, prompt_hat)
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        f_hat = self.cat_rec_syn_m(f_hat, x_encoder_out)
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
    
    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]
        
        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

       
        sos = cond_BD = self.class_emb(label_B)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            r_f_hat.add_(h_up*self.cat_rec_syn_p_ratio[si])

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                

        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 


@ARCH_REGISTRY.register()
class VarCodeFormer12(VarVQAutoEncoder2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                ):
        super(VarCodeFormer12, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)

        # transformer        
        self.n_layers =  n_layers       
        self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,
            connect_list = connect_list, # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder7(**ddconfig)
        

        # refuse rec and syn prompt
        self.cat_rec_syn_p = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4) #ResBlock(64, 32)
        self.cat_rec_syn_m = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4)
        self.cat_rec_syn_p_ratio = nn.Parameter(torch.ones(10))

        
        # # encoder res --> decoder

        self.connect_list = connect_list
        
        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(depth//3)
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        
        prompt_hat = self.cat_rec_syn_p(f_hat, prompt_hat)
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        f_hat = self.cat_rec_syn_m(f_hat, x_encoder_out)
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
    
    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]
        
        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

       
        sos = cond_BD = self.class_emb(label_B)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            r_f_hat.add_(h_up*self.cat_rec_syn_p_ratio[si])

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                

        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 


@ARCH_REGISTRY.register()
class VarCodeFormer13(VarVQAutoEncoder2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                ):
        super(VarCodeFormer13, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)

        # transformer        
        self.n_layers =  n_layers       
        self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)

        # main decoder
        ddconfig = dict(
            dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,
            connect_list = connect_list, # from vq-f16/config.yaml above
            
        )
        self.main_decoder = MainDecoder7(**ddconfig)
        

        # refuse rec and syn prompt
        self.cat_rec_syn_p = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4) #ResBlock(64, 32)
        self.cat_rec_syn_m = RcaMoudle(dim=32, input_resolution=(16,16), depth=2, num_heads=8, window_size=4)
        self.cat_rec_syn_p_ratio = nn.Parameter(torch.ones(10))

        
        # # encoder res --> decoder

        self.connect_list = connect_list
        
        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'32': 4, '64':3, '128':2, '256':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        self.cat_linear = nn.Linear(embed_dim*2, embed_dim)
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(depth//3)
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=True)
            


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    


    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, f_hat: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict: dict, fuse_list: List):
        
        prompt_hat = self.cat_rec_syn_p(f_hat, prompt_hat)
        h, dec_res_feats = self.decoder(self.post_quant_conv(prompt_hat))
        f_hat = self.cat_rec_syn_m(f_hat, x_encoder_out)
        x = self.main_decoder(dec_res_feats, f_hat, enc_feat_dict, fuse_list)
        return x.clamp_(-1, 1)
    
    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()

        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]
        
        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

       
        sos = cond_BD = self.class_emb(label_B)
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            r_f_hat.add_(h_up*self.cat_rec_syn_p_ratio[si])

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                

        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out, lq_feat, r_f_hat, enc_feat_dict, fuse_list).add_(1).mul_(0.5), lq_feat 


@ARCH_REGISTRY.register() 
class VarFormer14(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_2): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,s=16,k_s=3,
                ):
        super(VarFormer14, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums,s=s,k_s=k_s)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)
        


        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,s=s,k_s=k_s,
            
        )

        
        self.main_decoder = MainDecoder_varformer2_2(**ddconfig)
        

        
        
        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            
            self.copy_params(self.encoder, self.Mainencoder)
            
            
    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, enc_feat_dict_var: dict, fuse_list: List,inp_B3HW: torch.Tensor):
        
        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, enc_feat_dict_var=enc_feat_dict_var, fuse_list=fuse_list)
        return x.clamp_(-1, 1)+inp_B3HW
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict_var = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)

           

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        
        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat,enc_feat_dict_var=enc_feat_dict_var, out_feature_list=out_list)
        
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, enc_feat_dict_var=enc_feat_dict_var, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, enc_feat_dict_var = enc_feat_dict_var, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat 




@ARCH_REGISTRY.register() 
class VarFormer15(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2):
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,
                ):
        super(VarFormer15, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)
        


        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
            
        )

        
        self.main_decoder = MainDecoder_varformer2(**ddconfig)
        

        
        
        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # class-free r
        cfg=1.5
        r_ls = []
        
        
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            t = cfg * ratio
            r_ls.append(t)
        r_ls = torch.tensor(r_ls)
        self.cfg_r = nn.Parameter(r_ls)
        
        

        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            
            self.copy_params(self.encoder, self.Mainencoder)
            
            
    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, enc_feat_dict_var: dict, fuse_list: List,inp_B3HW: torch.Tensor):
        
        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, enc_feat_dict_var=enc_feat_dict_var, fuse_list=fuse_list)
        return x.clamp_(-1, 1)+inp_B3HW
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, enc_feat_dict_var = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            # t = cfg * ratio
            t = self.cfg_r[si]
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)

           

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        
        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat,enc_feat_dict_var=enc_feat_dict_var, out_feature_list=out_list)
        
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, enc_feat_dict_var=enc_feat_dict_var, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, enc_feat_dict_var = enc_feat_dict_var, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat 



@ARCH_REGISTRY.register() 
class VarFormer16(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,
                ):
        super(VarFormer16, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)
        


        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
            
        )

        
        self.main_decoder = MainDecoder_varformer2_kzj(**ddconfig)
        

        
        
        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}
        






        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat
         
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)
            

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            
            self.copy_params(self.encoder, self.Mainencoder)
            
            
    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, fuse_list: List,inp_B3HW: torch.Tensor):
        
        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list)
        return x.clamp_(-1, 1)+inp_B3HW
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, _ = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)

           

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        
        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat, out_feature_list=out_list)
        
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat 


@ARCH_REGISTRY.register() 
class VarFormer17(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj_down_1): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,
                ):
        super(VarFormer17, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        # transformer
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)


        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
        )


        self.main_decoder = MainDecoder_varformer2_kzj_down_1(**ddconfig)

        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}





        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            self.copy_params(self.encoder, self.Mainencoder)
            

    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, fuse_list: List,inp_B3HW: torch.Tensor):

        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list)
        return x.clamp_(-1, 1)+inp_B3HW
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]



        x_encoder_out, _ = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)





            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            #idx_N.reshape(B, ph*pw)
            if gt_in:
                idx_Bl_all.append(idx_Bl)
            

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)

           

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                
                

                
                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        
        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat, out_feature_list=out_list)
        
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat 



@ARCH_REGISTRY.register() 
class VarFormer18(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj_down_1): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,
                ):
        super(VarFormer18, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)

        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
        )


        self.main_decoder = MainDecoder_varformer2_kzj_down_1_hfuse(**ddconfig)


        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}





        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            self.copy_params(self.encoder, self.Mainencoder)
            

    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, fuse_list: List,inp_B3HW: torch.Tensor):
        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list)
        return x.clamp_(-1, 1)+inp_B3HW
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False, g_seed=620664) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]


        x_encoder_out, _ = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]
        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)



            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            if gt_in:
                idx_Bl_all.append(idx_Bl)

            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)


            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                

                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG


        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat, out_feature_list=out_list)
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat 




@ARCH_REGISTRY.register() 
class VarFormer19(VarVQAutoEncoder2__norm_noDecoder_2encoder2_varformer2_kzj_patial_gres): 
    def __init__(self, 
                fix_modules=['quantize','decoder','post_quant_conv'], 
                ch_mult=None,
                num_res_blocks=None,
                dropout=0.0,
                vqgan_path=None,
                var_path=None,
                model_path=None,
                connect_list=['16', '32', '64', '128'],
                num_classes=1000, depth=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
                attn_l2_norm=False,
                patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   
                flash_if_available=True, fused_if_available=True,
                var_force_dpth=7,
                n_layers = 4,
                if_enhance = True,
                ):
        super(VarFormer19, self).__init__(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=patch_nums)
        
       

        dpr = 0.1 * depth/24
        num_heads = depth
        embed_dim = depth * 64

        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        self.var_force_dpth = var_force_dpth
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        

        self.word_embed = nn.Linear(self.Cvae, self.C)


        # transformer
        self.n_layers =  n_layers   
        self.if_enhance = if_enhance
        if self.if_enhance:    
            self.ft_layers = EnhanceLayers(embed_dim=32, n_layers=self.n_layers)


        # main decoder
        ddconfig = dict(
           dropout=dropout, ch=160, z_channels=32,
            in_channels=3, ch_mult=ch_mult, num_res_blocks=num_res_blocks,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            connect_list = connect_list,
            resamp_with_conv=True,   # always True, removed.
        )


        self.main_decoder = MainDecoder_varformer2_kzj_patial_gres(**ddconfig)

        # # encoder res --> decoder

        self.connect_list = connect_list

        # # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'128':0, '64':1, '32':2, '16':3} 
        #  # after first residual block for > 16, before attn layer for ==16

        self.fuse_generator_block = {'16': 4, '32':3, '64':2, '128':1}



        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # add.cat

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )


        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionAR(block_idx, embed_dim,attn_l2_norm=attn_l2_norm)
            for block_idx in range(int(depth//3))
        ])


        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)


        if vqgan_path is not None:
            key = self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu'),strict=False)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
        if var_path is not None:
            load_net = torch.load(var_path, map_location='cpu')

            key = self.load_state_dict(
                torch.load(var_path, map_location='cpu'),strict=False)

        
        if model_path is not None:
            key = self.load_state_dict(
                torch.load(model_path, map_location='cpu'),strict=False)
            self.copy_params(self.encoder, self.Mainencoder)
            

    def copy_params(self,src_module, dest_module):
        dest_state_dict = dest_module.state_dict()
        src_state_dict = src_module.state_dict()

        for name, param in src_state_dict.items():
            if name in dest_state_dict:
                dest_state_dict[name].copy_(param)

    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def fhat_to_img(self, x_encoder_out: torch.Tensor, prompt_hat: torch.Tensor, enc_feat_dict_vf: dict, fuse_list: List,inp_B3HW: torch.Tensor):

        x = self.main_decoder(prompt_hat=prompt_hat, x_encoder_out=x_encoder_out, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list)
        return x.clamp_(-1, 1) 
    
    def img2idx(self, inp_B3HW,out_list=[2, 1, 0]):
        x_encoder_out, enc_feat_dict = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)   

        return  x_idx_Bl    

    def forward(self, inp_B3HW, code_only=False, pixel_l = False, cfg=1.5, gt_in = False) -> torch.Tensor:  # returns logits_BLV


        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        x_encoder_out, _ = self.img_to_encoder_out_get_f(inp_B3HW, out_list)
        
        if self.if_enhance:
            x_encoder_out = self.ft_layers(x_encoder_out)

        x_idx_Bl = self.encoder_out_to_idxBl(x_encoder_out)

        first_h_BChw, lq_feat, x_BLCv_wo_first_l = self.quantize.idxBl_to_var_input(x_idx_Bl)
        lq_feat = x_encoder_out + (lq_feat - x_encoder_out).detach()
        rng = self.rng
        B = x_BLCv_wo_first_l.shape[0]
        label_B = torch.full((B,), fill_value=self.num_classes).to(x_BLCv_wo_first_l.device) 

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        next_token_map2 = next_token_map
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        r_f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        for b in self.blocks: b.attn.kv_caching(True)

        logits_BlV_all = None
        idx_Bl_all = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment

            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward

            for index in range(int(len(self.blocks)//3)):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                x = self.cross_attn_blocks[index](next_token_map2,x)
                
            for index in range(int(len(self.blocks)//3),len(self.blocks),1):
                x = self.blocks[index](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)


            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV_all = logits_BlV.clone()
            else:
                logits_BlV_all = torch.cat([logits_BlV_all, logits_BlV], dim=1)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=1, top_p=1, num_samples=1)[:, :, 0]
            if gt_in:
                idx_Bl_all.append(idx_Bl)


            h_BChw = self.quantize.embedding(idx_Bl)   # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map, h_up = self.quantize.get_next_autoregressive_input_h(si, len(self.patch_nums), f_hat, h_BChw)
            if si == 0:
                r_f_hat = h_up.unsqueeze(1)
            else:
                r_f_hat = torch.cat([r_f_hat,h_up.unsqueeze(1)],dim=1)

            if si != self.num_stages_minus_1:   # prepare for next stage

                next_token_map2 = x_BLCv_wo_first_l[:,cur_L-1:cur_L-1 + self.patch_nums[si+1] ** 2]
                next_token_map2 = self.word_embed(next_token_map2) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]

                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                

                next_token_map2 = next_token_map2.repeat(2, 1, 1)
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG


        x_encoder_out2, enc_feat_dict_vf = self.img_to_mainencoder_out_get_f(x=inp_B3HW, prompt_hat=r_f_hat, out_feature_list=out_list)
        if gt_in:
            return idx_Bl_all
        if pixel_l:
            return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf=enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW), lq_feat, logits_BlV_all
        
        elif code_only: # for training stage II
            return lq_feat
        
        return self.fhat_to_img(x_encoder_out=x_encoder_out2, prompt_hat=r_f_hat, enc_feat_dict_vf = enc_feat_dict_vf, fuse_list=fuse_list,inp_B3HW=inp_B3HW).add_(1).mul_(0.5), lq_feat








