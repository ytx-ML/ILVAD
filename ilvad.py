import numpy as np
from typing import Optional
from transformers.models.llama.modeling_llama import *
from utils import get_layers
import torch
import functools


class EnhanceEvidenceAttention:
    def __init__(self, img_start, img_end, respond_start, enhanced_map, layers_to_modify, beta, last_layer_idx):
        self.s = img_start
        self.e = img_end
        self.r = respond_start
        self.enhanced_map = enhanced_map
        self.layers_to_modify = layers_to_modify
        self.beta = beta
        self.last_layer_idx = last_layer_idx
        self.tv_score = None
        self.tv_score_index = None
    

def llava_forward(
        self,
        ilvad: EnhanceEvidenceAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


    epsilon = 1e-8
    if attn_weights.shape[-2] > 1:
    
        if ilvad.tv_score_index is None:
            ilvad.tv_score_index = torch.zeros(attn_weights.shape[-2]-ilvad.e, device=ilvad.enhanced_map.device)

        for i in range(ilvad.e, attn_weights.shape[-2]-1):
            vision_attn = attn_weights[0, :, i, ilvad.s:ilvad.e]
            
            vision_scores = torch.sum(vision_attn * torch.log(ilvad.enhanced_map), dim=-1)/(torch.sum(vision_attn, dim=-1)+epsilon)

            num_heads_to_select = attn_weights.size(1) // 2
            selected_heads_vision = torch.topk(vision_scores, num_heads_to_select).indices
            selected_vision_attn = vision_attn[selected_heads_vision]
            avg_selected_vision_attn = torch.mean(selected_vision_attn,dim=0)
            
            current_token_score = torch.sum(torch.log(ilvad.enhanced_map) * avg_selected_vision_attn)
            ilvad.tv_score_index[i - ilvad.e] += current_token_score

    # Extract attention weights for vision and query tokens
    vision_attn = attn_weights[0, :, -1, ilvad.s:ilvad.e]
    query_attn = attn_weights[0, :, -1, ilvad.e:] #[32,...]

    # Select top attention heads for vision
    vision_scores = torch.sum(vision_attn * torch.log(ilvad.enhanced_map), dim=-1)/(torch.sum(vision_attn, dim=-1)+epsilon)

    num_heads_to_select = attn_weights.size(1) // 2
    selected_heads_vision = torch.topk(vision_scores, num_heads_to_select).indices
    selected_vision_attn = vision_attn[selected_heads_vision]
    avg_selected_vision_attn = torch.mean(selected_vision_attn,dim=0)


    if ilvad.tv_score is None:
        ilvad.tv_score = torch.sum(torch.log(ilvad.enhanced_map) * avg_selected_vision_attn)
    else:
        ilvad.tv_score += torch.sum(torch.log(ilvad.enhanced_map) * avg_selected_vision_attn)
    if self.layer_idx == ilvad.last_layer_idx:
        if ilvad.tv_score_index is None:
            ilvad.tv_score_index = ilvad.tv_score.unsqueeze(0)
        else:
            ilvad.tv_score_index = torch.cat([ilvad.tv_score_index, ilvad.tv_score.unsqueeze(0)])
        ilvad.tv_score = None
        

    if self.layer_idx in ilvad.layers_to_modify and ilvad.tv_score_index is not None:
        selected_vision_attn *= ilvad.enhanced_map
        vision_attn[selected_heads_vision] = selected_vision_attn

        query_scores = query_attn.sum(dim=-1)
        selected_heads_query = torch.topk(query_scores, num_heads_to_select).indices

        # normalize tv_score_index
        tv_min = ilvad.tv_score_index.min()
        tv_max = ilvad.tv_score_index.max()
        mean, std = ilvad.tv_score_index.mean(), ilvad.tv_score_index.std()
        lower, upper = mean - 3 * std, mean + 3 * std
        ilvad.tv_score_index = torch.clamp(ilvad.tv_score_index, min=lower.item(), max=upper.item())
    
        normalized_scores = ((ilvad.tv_score_index - tv_min) / (tv_max - tv_min)) + ilvad.beta 
        num_existing_tokens = normalized_scores.size(0)
        for head_idx in selected_heads_query:
            query_attn[head_idx, :num_existing_tokens] *= normalized_scores
        
        attn_weights[0, :, -1] = attn_weights[0, :, -1] / (attn_weights[0, :, -1].sum(dim=-1, keepdim=True) + epsilon)

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class AttentionEnhancerContext:
    def __init__(self, model, img_start, img_end, respond_start, enhanced_map, layers_to_modify, beta, model_name):
        self.model = model
        self.img_start = img_start
        self.img_end = img_end
        self.respond_start = respond_start
        self.enhanced_map = enhanced_map
        self.layers_to_modify = layers_to_modify
        self.beta = beta
        self.get_layers = get_layers
        self.model_name = model_name
        self.original_forwards = {}  
        self.ilvad = None  

    def __enter__(self):

        layers = self.get_layers(self.model)

        last_layer_idx = getattr(layers[-1].self_attn, "layer_idx", len(layers) - 1)

        self.ilvad = EnhanceEvidenceAttention(
            self.img_start, self.img_end, self.respond_start,
            self.enhanced_map, self.layers_to_modify, self.beta, last_layer_idx
        )


        for idx, layer in enumerate(self.get_layers(self.model)):
            self.original_forwards[idx] = layer.self_attn.forward
            new_forward = functools.partial(llava_forward, layer.self_attn, self.ilvad)
            layer.self_attn.forward = new_forward

        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        for idx, layer in enumerate(self.get_layers(self.model)):
            if idx in self.original_forwards:
                layer.self_attn.forward = self.original_forwards[idx]
        self.original_forwards.clear()
        self.ilvad = None

        return False