import transformers
from torch import nn
from typing import Unpack, Callable
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import rotate_half, eager_attention_forward

logger = logging.get_logger(__name__)

def apply_rotary_pos_emb(q, k, cos, sin, kv_cos=None, kv_sin=None, position_ids=None, kv_position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    if kv_cos is not None and kv_sin is not None:
        kv_cos = kv_cos.unsqueeze(unsqueeze_dim)
        kv_sin = kv_sin.unsqueeze(unsqueeze_dim)
        k_embed = (k * kv_cos) + (rotate_half(k) * kv_sin)
    else:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class T5LikeLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.n_heads = config.num_attention_heads
        self.key_value_proj_dim = config.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(self, 
                hidden_states,
                position_embeddings,
                attention_mask=None,
                key_value_states=None,
                kv_position_embeddings=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                cache_position=None,
                **kwargs: Unpack[FlashAttentionKwargs],
                ):
        batch_size, seq_length, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None
        
        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states

        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True
        
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_length, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        if kv_position_embeddings is not None:
            kv_cos, kv_sin = kv_position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, kv_cos, kv_sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
