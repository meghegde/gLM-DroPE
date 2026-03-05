import logging
from typing import Optional, Any, Type

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.models.esm.modeling_esm import EsmConfig, EsmAttention
from transformers import AutoModelForMaskedLM, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Existing attention wrappers
# -----------------------------

# NoPE attention wrapper
def nope_nt2(BaseAttentionClass: Type[nn.Module]) -> Type[nn.Module]:
    class NoPENTv2Attention(BaseAttentionClass):
        def __init__(self, config: EsmConfig, layer_idx: int = 0, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.layer_idx = layer_idx
            self.config = config

        def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
            output_attentions = kwargs.get("output_attentions", False)

            batch, seq_len, _ = hidden_states.shape
            num_heads = self.self.num_attention_heads
            head_dim = self.self.attention_head_size

            # Linear projections
            q = self.self.query(hidden_states)
            k = self.self.key(hidden_states)
            v = self.self.value(hidden_states)

            # Reshape for multi-head
            q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

            # Compute attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)

            # Combine heads
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
            attn_output = self.output.dense(attn_output)

            outputs = (attn_output,)
            if output_attentions:
                outputs = outputs + (attn_probs,)

            return outputs

        @classmethod
        def from_source(cls, source_module: nn.Module, config: EsmConfig, layer_idx: int = 0):
            new_module = cls(config=config, layer_idx=layer_idx)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            return new_module

    NoPENTv2Attention.__name__ = f"NoPE{BaseAttentionClass.__name__}"
    return NoPENTv2Attention

# Q/K Norm + NoPE attention wrapper
def qk_norm_nt2(BaseAttentionClass: type) -> type:
    class QKNormNoPENTv2Attention(BaseAttentionClass):
        def __init__(self, config: EsmConfig, layer_idx: int = 0, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.layer_idx = layer_idx

            # RMSNorm over hidden_size (heads flattened)
            self.q_norm = nn.RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-5),
            )
            self.k_norm = nn.RMSNorm(
                config.hidden_size,
                eps=getattr(config, "rms_norm_eps", 1e-5),
            )

        forward = nt2_qk_norm_nope_forward

        @classmethod
        def from_source(cls, source_module: nn.Module, config: EsmConfig, layer_idx: int = 0):
            new_module = cls(config=config, layer_idx=layer_idx)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            return new_module

    QKNormNoPENTv2Attention.__name__ = f"QKNormNoPE{BaseAttentionClass.__name__}"
    return QKNormNoPENTv2Attention

# Custom Q/K Norm forward function
def nt2_qk_norm_nope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    layer_past=None,
    output_attentions: bool = False,
    **kwargs
):
    batch, seq_len, _ = hidden_states.shape

    num_heads = self.self.num_attention_heads
    head_dim = self.self.attention_head_size

    q = self.self.query(hidden_states)
    k = self.self.key(hidden_states)
    v = self.self.value(hidden_states)

    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    # RMSNorm over hidden_size
    q_flat = q.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    k_flat = k.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    q = self.q_norm(q_flat).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = self.k_norm(k_flat).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    # Attention scores
    attn_scores = torch.matmul(q, k.transpose(-1, -2))
    attn_scores = attn_scores / (head_dim ** 0.5)

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.self.dropout(attn_probs)

    attn_output = torch.matmul(attn_probs, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)
    attn_output = self.output.dense(attn_output)

    outputs = (attn_output,)
    if output_attentions:
        outputs = outputs + (attn_probs,)

    return outputs

# Initialize attention classes
NoPENTv2Attention = nope_nt2(EsmAttention)
QKNormNoPENTv2Attention = qk_norm_nt2(EsmAttention)

# -----------------------------
# Model loader with classification loss
# -----------------------------
class DROPEClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels=2):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()  # scalar by default

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, use_cache=False) # Disable KV cache

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return (loss, logits) if loss is not None else logits

def create_drope_ntv2_model(base_model_name_or_path: str, attention_type: str = "nope", num_labels: int = 2):
    """
    Load NT-v2, patch attention layers, and wrap in DROPEClassificationModel.
    """
    # Load base model
    config = EsmConfig.from_pretrained(base_model_name_or_path)
    config.use_cache = False # Disable KV cache to save memory
    base_model = AutoModel.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Patch attention layers
    for i, layer in enumerate(base_model.encoder.layer):
        if hasattr(layer, "attention"):
            orig_attn = layer.attention
            if attention_type == "nope":
                AttnClass = NoPENTv2Attention
            elif attention_type == "qk_norm_nope":
                AttnClass = QKNormNoPENTv2Attention
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")

            layer.attention = AttnClass.from_source(orig_attn, config=config, layer_idx=i)
            logger.info(f"Patched NT-v2 attention layer {i} with {AttnClass.__name__}")

    # Wrap in classification model
    model = DROPEClassificationModel(base_model, num_labels=num_labels)
    return model

def create_drope_ntv2_mlm_model(
    base_model_name_or_path: str,
    attention_type: str = "nope",
    output_hidden_states: bool = False
):
    config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
    )

    if output_hidden_states==True:
        # Enable hidden states in the config
        config.output_hidden_states = True

    model = AutoModelForMaskedLM.from_pretrained(
        base_model_name_or_path,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )

    # Patch attention layers
    for i, layer in enumerate(model.esm.encoder.layer):
        orig_attn = layer.attention

        if attention_type == "nope":
            AttnClass = NoPENTv2Attention
        elif attention_type == "qk_norm_nope":
            AttnClass = QKNormNoPENTv2Attention
        else:
            raise ValueError(attention_type)

        layer.attention = AttnClass.from_source(
            orig_attn,
            config=config,
            layer_idx=i,
        )

    return model