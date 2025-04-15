import math
from copy import deepcopy
from typing import Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.model import CLIP
from clip.model import ModifiedResNet as CLIP_RN
from clip.model import VisionTransformer as CLIP_ViT
from clip.model import ResidualAttentionBlock as CLIP_Block

from timm.models.resnet import ResNet as TIMM_RN
from timm.models.vision_transformer import VisionTransformer as TIMM_ViT
from timm.models.vision_transformer import Block as TIMM_Block

from .modules import *
from .classifiers import *


def create_linear(weight: nn.Parameter, bias: nn.Parameter) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0])
    linear.weight = weight
    linear.bias = bias
    return linear


def convert_layernorm(ln: nn.LayerNorm) -> NonHalfLayerNorm:
    nonhalf_ln = NonHalfLayerNorm(ln.normalized_shape, ln.eps, ln.elementwise_affine, bias=(ln.bias is not None))
    if ln.weight is not None:
        nonhalf_ln.weight = NonHalfParameter(ln.weight.float())
    if ln.bias is not None:
        nonhalf_ln.bias = NonHalfParameter(ln.bias.float())
    return nonhalf_ln


class PEFT_Block(nn.Module):
    def __init__(self, block: Union[CLIP_Block, TIMM_Block], is_causal=False, norm_first=True):
        super().__init__()

        if isinstance(block, CLIP_Block):
            self.norm1 = convert_layernorm(block.ln_1)
            self.attn_in = create_linear(block.attn.in_proj_weight, block.attn.in_proj_bias)
            self.attn_out = block.attn.out_proj
            self.norm2 = convert_layernorm(block.ln_2)
            self.mlp_in = block.mlp[0]
            self.mlp_act = block.mlp[1]
            self.mlp_out = block.mlp[2]
            self.num_heads = block.attn.num_heads

        elif isinstance(block, TIMM_Block):
            self.norm1 = convert_layernorm(block.norm1)
            self.attn_in = block.attn.qkv
            self.attn_out = block.attn.proj
            self.norm2 = convert_layernorm(block.norm2)
            self.mlp_in = block.mlp.fc1
            self.mlp_act = block.mlp.act
            self.mlp_out = block.mlp.fc2
            self.num_heads = block.attn.num_heads

        else:
            raise TypeError

        self.is_causal = is_causal
        self.norm_first = norm_first

        self.tuner = nn.ParameterDict()
    
    @property
    def embed_dim(self):
        return self.attn_in.weight.shape[1]

    @property
    def head_dim(self):
        return self.embed_dim // self.num_heads

    @property
    def dtype(self):
        return self.attn_in.weight.dtype

    @property
    def device(self):
        return self.attn_in.weight.device
    
    def unfreeze_params(self):
        for name, param in self.named_parameters():
            self.tuner[name.replace(".", "_")] = param  # to be optimized
    
    def unfreeze_bias(self):
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                self.tuner[name.replace(".", "_")] = param  # to be optimized

    def add_learnable_prompt(self, prompt_len=0):
        prompt = nn.Parameter(torch.empty(prompt_len, self.embed_dim, dtype=self.dtype, device=self.device))
        nn.init.normal_(prompt, std=0.02)

        self.tuner["prompt"] = prompt  # to be optimized

    def add_lora(self, bottle_dim):
        self.tuner["lora"] = nn.ModuleDict({
            "q": LoRA(self.embed_dim, bottle_dim, dtype=self.dtype, device=self.device),
            "v": LoRA(self.embed_dim, bottle_dim, dtype=self.dtype, device=self.device),
        })  # to be optimized

    def add_adapter(self, bottle_dim):
        self.tuner["adapter"] = Adapter(self.embed_dim, bottle_dim, dtype=self.dtype, device=self.device)  # to be optimized
    
    def add_adaptformer(self, bottle_dim):
        self.tuner["adaptformer"] = AdaptFormer(self.embed_dim, bottle_dim, dtype=self.dtype, device=self.device)  # to be optimized

    def add_ssf(self):
        self.tuner["ssf"] = nn.ParameterDict()
        self.tuner["ssf"]["attn_in"] = SSF(self.attn_in.weight.shape[0], dtype=self.dtype, device=self.device)  # to be optimized
        self.tuner["ssf"]["attn_out"] = SSF(self.attn_out.weight.shape[0], dtype=self.dtype, device=self.device)  # to be optimized
        self.tuner["ssf"]["mlp_in"] = SSF(self.mlp_in.weight.shape[0], dtype=self.dtype, device=self.device)  # to be optimized
        self.tuner["ssf"]["mlp_out"] = SSF(self.mlp_out.weight.shape[0], dtype=self.dtype, device=self.device)  # to be optimized

    def add_aft(self, ratio, loc="mlp", seed=None):
        generator = torch.Generator().manual_seed(seed)  # fix the generator
        def _generate_mask(input):
            num_params = input.numel()
            num_masked = int(num_params * ratio)
            masked_indexs = torch.randperm(num_params, generator=generator, dtype=torch.int64)[:num_masked]
            mask = torch.zeros(num_params, dtype=bool).scatter_(dim=0, index=masked_indexs, value=True).view(input.shape)
            return mask.to(self.device)

        self.tuner["aft"] = nn.ParameterDict()
        if loc in ("attn", "all"):
            self.tuner["aft"]["attn_in"] = MaskedLinear(self.attn_in.weight, self.attn_in.bias, _generate_mask(self.attn_in.weight), _generate_mask(self.attn_in.bias))  # to be optimized
            self.tuner["aft"]["attn_out"] = MaskedLinear(self.attn_out.weight, self.attn_out.bias, _generate_mask(self.attn_out.weight), _generate_mask(self.attn_out.bias))  # to be optimized
        if loc in ("mlp", "all"):
            self.tuner["aft"]["mlp_in"] = MaskedLinear(self.mlp_in.weight, self.mlp_in.bias, _generate_mask(self.mlp_in.weight), _generate_mask(self.mlp_in.bias))  # to be optimized
            self.tuner["aft"]["mlp_out"] = MaskedLinear(self.mlp_out.weight, self.mlp_out.bias, _generate_mask(self.mlp_out.weight), _generate_mask(self.mlp_out.bias))  # to be optimized

    def forward(self, x):
        _batch_size = x.shape[1]
        _embed_dim = x.shape[2]

        if hasattr(self.tuner, "prompt"):
            raw_seq_len = self.raw_input_shape[0]
            prefix = x[:1, :, :]
            suffix = x[-(raw_seq_len-1):, :, :]
            prompt = self.tuner.prompt.unsqueeze(1).expand(-1, _batch_size, -1)
            x = torch.cat([prefix, prompt, suffix], dim=0)

        _seq_len_now = x.shape[0]

        ###############################
        ## Multi-Head Self-Attention ##
        ###############################
        identity = x

        if self.norm_first:
            x = self.norm1(x)

        if hasattr(self.tuner, "aft") and hasattr(self.tuner.aft, "attn_in"):
            qkv = self.tuner.aft.attn_in(x)
        else:
            qkv = self.attn_in(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if hasattr(self.tuner, "lora"):
            q = q + self.tuner.lora.q(x)
            v = v + self.tuner.lora.v(x)
        
        if hasattr(self.tuner, "ssf") and hasattr(self.tuner.ssf, "attn_in"):
            qkv = torch.cat([q, k, v], dim=-1)
            qkv = self.tuner.ssf.attn_in(qkv)
            q, k, v = qkv.chunk(3, dim=-1)

        q = q.contiguous().view(q.shape[0], q.shape[1] * self.num_heads, self.head_dim).transpose(0, 1)  # [bsz * num_heads, seq_len, head_dim]
        k = k.contiguous().view(k.shape[0], k.shape[1] * self.num_heads, self.head_dim).transpose(0, 1)  # [bsz * num_heads, seq_len, head_dim]
        v = v.contiguous().view(v.shape[0], v.shape[1] * self.num_heads, self.head_dim).transpose(0, 1)  # [bsz * num_heads, seq_len, head_dim]

        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        # scaled_dot_product_attention:
        # attn = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # if self.is_causal:
        #     attn += torch.empty(_seq_len_now, _seq_len_now, dtype=self.dtype, device=self.device).fill_(float("-inf")).triu_(1)
        # attn = F.softmax(attn, dim=-1)
        # x = attn @ v

        x = x.transpose(0, 1).contiguous().view(-1, _embed_dim)  # [seq_len * bsz * num_heads, head_dim]
        
        if hasattr(self.tuner, "aft") and hasattr(self.tuner.aft, "attn_out"):
            x = self.tuner.aft.attn_out(x)
        else:
            x = self.attn_out(x)
        
        if hasattr(self.tuner, "ssf") and hasattr(self.tuner.ssf, "attn_out"):
            x = self.tuner.ssf.attn_out(x)

        x = x.view(_seq_len_now, _batch_size, _embed_dim)
        
        x = identity + x

        if not self.norm_first:
            x = self.norm1(x)

        ##########################
        ## Feed-Forward Network ##
        ##########################
        identity = x

        if self.norm_first:
            x = self.norm2(x)
        
        if hasattr(self.tuner, "aft") and hasattr(self.tuner.aft, "mlp_in"):
            x = self.tuner.aft.mlp_in(x)
        else:
            x = self.mlp_in(x)
        
        if hasattr(self.tuner, "ssf") and hasattr(self.tuner.ssf, "mlp_in"):
            x = self.tuner.ssf.mlp_in(x)

        x = self.mlp_act(x)

        if hasattr(self.tuner, "aft") and hasattr(self.tuner.aft, "mlp_out"):
            x = self.tuner.aft.mlp_out(x)
        else:
            x = self.mlp_out(x)

        if hasattr(self.tuner, "ssf") and hasattr(self.tuner.ssf, "mlp_out"):
            x = self.tuner.ssf.mlp_out(x)
        
        if hasattr(self.tuner, "adapter"):
            x = x + self.tuner.adapter(x)
        
        if hasattr(self.tuner, "adaptformer"):
            x = x + self.tuner.adaptformer(identity)

        x = identity + x

        if not self.norm_first:
            x = self.norm2(x)
        
        return x
        

class PEFT_Transformer(nn.Module):
    def __init__(self, blocks: nn.Sequential, **kwargs):
        super().__init__()

        self.blocks = nn.ModuleList([PEFT_Block(block, **kwargs) for block in blocks])

        self.tuner = nn.ParameterDict({
            "blocks": nn.ModuleList([block.tuner for block in self.blocks])
        })

    @property
    def dtype(self):
        return self.blocks[0].dtype

    @property
    def device(self):
        return self.blocks[0].device
    
    def unfreeze_params(self, layers=None):
        for i in layers:
            self.blocks[i].unfreeze_params()

    def unfreeze_bias(self, layers=None):
        for i in layers:
            self.blocks[i].unfreeze_bias()

    def add_learnable_prompt(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_learnable_prompt(**kwargs)

    def add_lora(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_lora(**kwargs)

    def add_adapter(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_adapter(**kwargs)
    
    def add_adaptformer(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_adaptformer(**kwargs)

    def add_ssf(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_ssf(**kwargs)

    def add_aft(self, layers=None, **kwargs):
        for i in layers:
            self.blocks[i].add_aft(**kwargs)
    
    def forward(self, x, **kwargs):
        raw_input_shape = torch.tensor(x.shape)
        for block in self.blocks:
            block.register_buffer("raw_input_shape", raw_input_shape, persistent=False)
            x = block(x, **kwargs)
        return x


class PEFT_ViT(PEFT_Transformer):
    def __init__(self, model: Union[CLIP_ViT, TIMM_ViT]):

        if isinstance(model, CLIP_ViT):
            super().__init__(model.transformer.resblocks)
            self.patch_embedding = model.conv1
            self.class_embedding = model.class_embedding
            self.positional_embedding = model.positional_embedding
            self.norm_pre = convert_layernorm(model.ln_pre)
            self.norm_post = convert_layernorm(model.ln_post)
    
        elif isinstance(model, TIMM_ViT):
            super().__init__(model.blocks)
            self.patch_embedding = model.patch_embed.proj
            self.class_embedding = model.cls_token
            self.positional_embedding = model.pos_embed
            self.norm_pre = nn.Identity()
            self.norm_post = convert_layernorm(model.norm)

        else:
            raise TypeError
    
    @property
    def embed_dim(self):
        return self.patch_embedding.weight.shape[0]

    def forward(self, image):
        x = self.patch_embedding(image.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.type(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.type(x.dtype)
        x = self.norm_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = super().forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.norm_post(x)[:, 0, :]

        return x


class PEFT_Text(PEFT_Transformer):
    def __init__(self, model: Union[CLIP]):

        if isinstance(model, CLIP):
            super().__init__(model.transformer.resblocks, is_causal=True)
            self.token_embedding = model.token_embedding
            self.positional_embedding = model.positional_embedding
            self.norm_final = convert_layernorm(model.ln_final)
        
        else:
            raise TypeError
    
    @property
    def embed_dim(self):
        return self.token_embedding.weight.shape[1]
    
    def _get_context_indices(self, text, relative_loc):
        sot_idx, eot_idx = 0, text.argmax(dim=-1, keepdim=True)
        txt_len = eot_idx - (sot_idx + 1)  # [batch_size, 1] or [1]
        absolute_idx = relative_loc % txt_len + (sot_idx + 1)  # [batch_size, ctx_len] or [ctx_len]
        return absolute_idx
    
    def add_learnable_context(self, ctx_loc, init_text=None):
        """ ctx_loc: the relative location of context in the text.
        For example, ctx_loc = [0, 1, -2, -1] means the "X" in "X X {} X X" is the learned context.
        """
        ctx_len = len(ctx_loc)
        ctx_vector = nn.Parameter(torch.empty(ctx_len, self.embed_dim, dtype=self.dtype, device=self.device))
        ctx_loc = torch.tensor(ctx_loc, dtype=torch.int64, device=self.device)

        if init_text is None:
            nn.init.normal_(ctx_vector, std=0.02)
        else:
            ctx_idx = self._get_context_indices(init_text, ctx_loc)
            with torch.no_grad():
                x = self.token_embedding(init_text).type(self.dtype)  # [seq_len]
            ctx_vector.data = x[ctx_idx]
        
        self.tuner["ctx"] = ctx_vector  # to be optimized
        self.register_buffer("ctx_loc", ctx_loc)

    def forward(self, text):
        """ text: the tokenized text, shape=[batch_size, seq_len=77],
        e.g. [[49406 (sot), 1981, 269, 49407 (eot), 0, 0, ... 0]]
        """
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, seq_len, embed_dim]

        # learnable context
        if hasattr(self.tuner, "ctx"):
            ctx_idx = self._get_context_indices(text, self.ctx_loc)
            # replace the original context with learnable context
            x[torch.arange(x.shape[0]).unsqueeze(1).expand_as(ctx_idx), ctx_idx] = self.tuner.ctx.expand(x.shape[0], -1, -1)

        x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = super().forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.norm_final(x).type(self.dtype)

        eot_offset = x.shape[1] - text.shape[1]

        # x.shape = [batch_size, seq_len, transformer.width]
        # take feature from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1) + eot_offset]

        return x


class PEFT_RN(nn.Module):
    def __init__(self, model: Union[CLIP_RN, TIMM_RN]):
        super().__init__()

        if isinstance(model, CLIP_RN):
            self.stem = nn.Sequential(OrderedDict({
                "conv1": model.conv1, "bn1": model.bn1, "relu1": model.relu1,
                "conv2": model.conv2, "bn2": model.bn2, "relu2": model.relu2,
                "conv3": model.conv3, "bn3": model.bn3, "relu3": model.relu3,
                "avgpool": model.avgpool,
            }))
            self.blocks = nn.Sequential(
                *model.layer1,
                *model.layer2,
                *model.layer3,
                *model.layer4,
            )
            self.global_pool = model.attnpool
            self.embed_dim = model.output_dim
        
        elif isinstance(model, TIMM_RN):
            self.stem = nn.Sequential(OrderedDict({
                "conv1": model.conv1, "bn1": model.bn1, "act1": model.act1,
                "maxpool": model.maxpool,
            }))
            self.blocks = nn.Sequential(
                *model.layer1,
                *model.layer2,
                *model.layer3,
                *model.layer4,
            )
            self.global_pool = model.global_pool
            self.embed_dim = model.num_features

        else:
            raise TypeError

        self.tuner = nn.ParameterDict({
            "blocks": nn.ParameterDict()
        })

    @property
    def dtype(self):
        return self.stem[0].weight.dtype

    @property
    def device(self):
        return self.stem[0].weight.device
    
    def unfreeze_params(self, **kwargs):
        for name, param in self.blocks.named_parameters():
            self.tuner["blocks"][name.replace(".", "_")] = param  # to be optimized
    
    def unfreeze_bias(self, **kwargs):
        for name, param in self.blocks.named_parameters():
            if name.endswith("bias"):
                self.tuner["blocks"][name.replace(".", "_")] = param  # to be optimized
    
    def unfreeze_bn(self, **kwargs):
        for name, module in self.blocks.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.tuner["blocks"][name.replace(".", "_")] = module  # to be optimized

    def add_ssf(self, **kwargs):
        self.tuner["ssf"] = SSF(self.embed_dim, dtype=self.dtype, device=self.device)  # to be optimized

    def forward(self, image):
        x = image.type(self.dtype)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        
        if hasattr(self.tuner, "ssf"):
            x = self.tuner.ssf(x, dim=1)

        return x


class PEFT_Model(nn.Module):
    def __init__(self, model: Union[CLIP, TIMM_ViT]):
        super().__init__()

        if isinstance(model, CLIP):
            if isinstance(model.visual, CLIP_ViT):
                self.image_encoder = PEFT_ViT(model.visual)
                self.image_proj = model.visual.proj
            elif isinstance(model.visual, CLIP_RN):
                self.image_encoder = PEFT_RN(model.visual)
            self.text_encoder = PEFT_Text(model)
            self.text_proj = model.text_projection
            self.logit_scale = model.logit_scale
        
        elif isinstance(model, TIMM_ViT):
            self.image_encoder = PEFT_ViT(model)
        elif isinstance(model, TIMM_RN):
            self.image_encoder = PEFT_RN(model)
        
        else:
            raise TypeError

        self.tuner = nn.ParameterDict({
            "image_encoder": self.image_encoder.tuner,
        })
        
        if hasattr(self, "text_encoder"):
            self.tuner["text_encoder"] = self.text_encoder.tuner
    
    @property
    def dtype(self):
        return self.image_encoder.dtype
    
    @property
    def device(self):
        return self.image_encoder.device

    def unfreeze_image_proj(self):
        self.tuner["image_proj"] = self.image_proj  # to be optimized
    
    def unfreeze_text_proj(self):
        self.tuner["text_proj"] = self.text_proj  # to be optimized
    
    def unfreeze_logit_scale(self):
        self.tuner["logit_scale"] = self.logit_scale  # to be optimized
    
    def add_clip_adapter(self, bottle_dim, image=True, text=True):
        if image:
            self.tuner["image_adapter"] = Adapter(self.image_proj.shape[1], bottle_dim=bottle_dim, dtype=self.dtype, device=self.device)
        if text:
            self.tuner["text_adapter"] = Adapter(self.text_proj.shape[1], bottle_dim=bottle_dim, dtype=self.dtype, device=self.device)

    def add_classifier(self, classifier_type, num_classes, **kwargs):
        classifier = eval(classifier_type)(self.image_encoder.embed_dim, num_classes, dtype=self.dtype, device=self.device, **kwargs)
        self.tuner["classifier"] = classifier  # to be optimized
    
    def init_classifier_weight(self, class_features, feature_modality):
        if feature_modality == "text":
            class_features = F.normalize(class_features @ self.text_proj.data)
            if hasattr(self, "image_proj"):
                class_features = F.normalize(class_features @ self.image_proj.data.t())
        elif feature_modality == "image":
            class_features = F.normalize(class_features)
        else:
            raise ValueError
        self.tuner.classifier.weight.data = class_features

    def forward(self, image, text=None, is_text_feature=False, use_classifier=False, return_feature=False):
        image_feature = self.image_encoder(image)

        if return_feature:
            return image_feature
        
        if use_classifier:
            logit = self.tuner.classifier(image_feature)
        else:
            if hasattr(self, "image_proj"):
                image_feature = image_feature @ self.image_proj
            
            text_feature = text if is_text_feature else self.text_encoder(text)
            text_feature = text_feature @ self.text_proj

            if hasattr(self.tuner, "image_adapter"):
                image_feature = image_feature + self.tuner.image_adapter(image_feature)
            if hasattr(self.tuner, "text_adapter"):
                text_feature = text_feature + self.tuner.text_adapter(text_feature)

            logit = self.logit_scale.exp() * F.normalize(image_feature) @ F.normalize(text_feature).t()
        
        return logit

    def convert_to_fp16(self):
        """Convert applicable model parameters to fp16"""
        
        def _apply_fn(module: torch.nn.Module, fn, skipped_type=()):
            for child in module.children():
                if isinstance(child, skipped_type):
                    continue
                _apply_fn(child, fn, skipped_type)

            module._apply(fn, recurse=False)
        
        fn = lambda t: t.half() if t.is_floating_point() else t
        _apply_fn(self, fn, skipped_type=(NonHalfLayerNorm))