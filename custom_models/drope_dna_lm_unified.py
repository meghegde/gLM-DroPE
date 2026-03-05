import logging
from typing import Type
import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from transformers.models.esm.modeling_esm import EsmAttention
from transformers.models.big_bird.modeling_big_bird import BigBirdSelfAttention


# =========================================================
# Logging
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# Stable QK-Norm module (per-head RMSNorm)
# =========================================================

class PerHeadRMSNorm(nn.Module):
    """
    RMSNorm applied independently to each attention head.
    Shape expected: [batch, heads, seq, head_dim]
    """
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, H, S, D]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.scale


# =========================================================
# ESM Stable Wrapper
# =========================================================

def build_stable_esm_wrapper(attention_type: str):

    class StableESMAttention(EsmAttention):

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer_idx = layer_idx

            if attention_type == "qk_norm_nope":
                head_dim = self.self.attention_head_size
                self.q_norm = PerHeadRMSNorm(head_dim)
                self.k_norm = PerHeadRMSNorm(head_dim)

        def forward(self, hidden_states, attention_mask=None, output_attentions=False):

            batch, seq_len, _ = hidden_states.shape
            num_heads = self.self.num_attention_heads
            head_dim = self.self.attention_head_size

            # Standard projections
            q = self.self.query(hidden_states)
            k = self.self.key(hidden_states)
            v = self.self.value(hidden_states)

            # Reshape
            q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

            # Disable RoPE if needed
            if hasattr(self, "rotary_emb") and attention_type != "original":
                pass  # simply skip rotary call

            # Apply stable QK norm
            if attention_type == "qk_norm_nope":
                q = self.q_norm(q)
                k = self.k_norm(k)

            # Attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2))
            attn_scores = attn_scores / (head_dim ** 0.5)

            if attention_mask is not None:
                attn_scores += attention_mask

            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.self.dropout(attn_probs)

            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch, seq_len, num_heads * head_dim)

            attn_output = self.output.dense(attn_output)

            outputs = (attn_output,)
            if output_attentions:
                outputs += (attn_probs,)
            return outputs

        @classmethod
        def from_source(cls, source_module, config, layer_idx=0):
            new_module = cls(config=config, layer_idx=layer_idx)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            return new_module

    return StableESMAttention


# =========================================================
# BigBird Stable Wrapper (Sparse Safe)
# =========================================================

def build_stable_bigbird_wrapper(attention_type: str):

    class StableBigBirdAttention(BigBirdSelfAttention):

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer_idx = layer_idx

            if attention_type == "qk_norm_nope":
                head_dim = self.attention_head_size
                self.q_norm = PerHeadRMSNorm(head_dim)
                self.k_norm = PerHeadRMSNorm(head_dim)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            **kwargs
        ):

            # Projections
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)

            batch, seq_len, _ = query_layer.shape
            num_heads = self.num_attention_heads
            head_dim = self.attention_head_size

            query_layer = query_layer.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            key_layer = key_layer.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            value_layer = value_layer.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

            # Skip RoPE if present
            if hasattr(self, "rotary_emb") and attention_type != "original":
                pass

            if attention_type == "qk_norm_nope":
                query_layer = self.q_norm(query_layer)
                key_layer = self.k_norm(key_layer)

            # Now call original sparse attention computation
            return self.bigbird_block_sparse_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                head_mask,
                output_attentions,
            )

        @classmethod
        def from_source(cls, source_module, config, layer_idx=0):
            new_module = cls(config=config, layer_idx=layer_idx)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            return new_module

    return StableBigBirdAttention


# =========================================================
# Unified Loader
# =========================================================

def create_stable_unified_model(
    base_model_name: str,
    attention_type: str = "original",
):

    config = AutoConfig.from_pretrained(base_model_name)
    config.use_cache = False

    model = AutoModel.from_pretrained(
        base_model_name,
        config=config,
        trust_remote_code=True,
    )

    if hasattr(model, "esm"):
        Wrapper = build_stable_esm_wrapper(attention_type)
        for i, layer in enumerate(model.esm.encoder.layer):
            layer.attention = Wrapper.from_source(layer.attention, config, i)

    elif hasattr(model, "bigbird"):
        Wrapper = build_stable_bigbird_wrapper(attention_type)
        for i, layer in enumerate(model.bigbird.encoder.layer):
            layer.attention.self = Wrapper.from_source(layer.attention.self, config, i)

    else:
        raise ValueError("Unsupported architecture")

    return model