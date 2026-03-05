import torch
import torch.nn as nn
import inspect
import logging


from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    eager_attention_forward as llama_eager_attention_forward,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    eager_attention_forward as qwen2_eager_attention_forward,
    apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
)

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, sdpa_attention_forward

from typing import Type, Callable, Optional, Any

# Configure logger
logger = logging.getLogger(__name__)


def nope(BaseAttentionClass: Type[nn.Module]) -> Type[nn.Module]:
    """
    A factory function that creates a new attention class with RoPE disabled.

    This function takes a Hugging Face attention module class (e.g., LlamaAttention)
    and returns a new class that inherits from it. The new class overrides the
    `forward` method to nullify the effect of RoPE before calling the original
    implementation.

    Args:
        BaseAttentionClass: The base attention class to modify (e.g., LlamaAttention).

    Returns:
        A new class with RoPE functionality removed.
    """

    class NoPEAttention(BaseAttentionClass):
        """
        An attention module wrapper that effectively disables RoPE by modifying
        its inputs during the forward pass.
        """

        def forward(self, *args, **kwargs):
            # Bind the provided arguments to the base class's forward method signature.
            # This allows us to safely find and modify RoPE-related arguments by name.
            signature = inspect.signature(super().forward)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Strategy 1: For models like Llama that pass pre-computed embeddings.
            # Nullify RoPE by setting cos=1 and sin=0, which is an identity rotation.
            if "position_embeddings" in bound_args.arguments:
                cos, sin = bound_args.arguments["position_embeddings"]
                bound_args.arguments["position_embeddings"] = (
                    torch.ones_like(cos),
                    torch.zeros_like(sin),
                )
                logger.debug("Removed RoPE by modifying 'position_embeddings'.")

            else:
                raise NotImplementedError(
                    "removing RoPE is only supported for models that pass position_embeddings."
                )

            # Call the original forward method with the modified arguments.
            return super().forward(*bound_args.args, **bound_args.kwargs)

        @classmethod
        def from_source(
            cls, source_module: BaseAttentionClass, config: Any
        ) -> "NoPEAttention":
            """
            Creates a new NoPEAttention instance from an existing attention module,
            copying its configuration and weights.
            """
            # Instantiate the new module with the same config
            config = source_module.config
            layer_idx = source_module.layer_idx

            # TODO: This is a bit hacky, we assume (1) that the attention module has a layer_idx attribute.
            # (2) that the attention module has an o_proj attribute, and (3) that the attention module only
            # takes config and layer_idx as arguments.
            new_module = cls(config=config, layer_idx=layer_idx).to(
                source_module.o_proj.weight.device
            )

            # Copy the weights and buffers
            new_module.load_state_dict(source_module.state_dict())
            if hasattr(config, "softmax_scale"):
                print(f"Setting softmax scale to {config.softmax_scale}")
                new_module.scaling = config.softmax_scale

            return new_module

    # Set a descriptive name for the dynamically created class
    NoPEAttention.__name__ = f"NoPE{BaseAttentionClass.__name__}"
    return NoPEAttention


def qk_norm(
    BaseAttentionClass: Type[nn.Module], new_forward_fn: Callable
) -> Type[nn.Module]:
    """
    A factory to create a customized attention class. It returns a class where the
    entire `forward` method is replaced by the provided function. The provided
    function MUST accept `self` as its first argument to access the module's
    parameters (e.g., self.q_proj).

    Args:
        BaseAttentionClass: The base attention class to modify (e.g., LlamaAttention).
        new_forward_fn: A function to replace the entire forward pass.
    Returns:
        A new, customized attention class.
    """

    class QKNormAttention(BaseAttentionClass):
        def __init__(self, *args, **kwargs):
            # Extract config from args or kwargs to use before super().__init__
            signature = inspect.signature(super().__init__)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            config = bound_args.arguments["config"]

            super().__init__(*args, **kwargs)
            # Create the norm functions
            self.q_norm = nn.RMSNorm(
                config.num_attention_heads * config.head_dim, config.rms_norm_eps
            )
            self.k_norm = nn.RMSNorm(
                config.num_key_value_heads * config.head_dim, config.rms_norm_eps
            )
            # Bind the qk_norm forward to the instance.
            self.forward = new_forward_fn.__get__(self, self.__class__)

        @classmethod
        def from_source(cls, source_module: BaseAttentionClass, model_config: Any) -> "QKNormAttention":
            config = source_module.config
            # Determine the correct device from one of the module's parameters
            device = next(source_module.parameters()).device
            new_module = cls(config, source_module.layer_idx).to(device)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            if hasattr(model_config, "softmax_scale"):
                print(f"Setting softmax scale to {model_config.softmax_scale}")
                new_module.scaling = config.softmax_scale
            return new_module

    QKNormAttention.__name__ = f"QKNormNoPE{BaseAttentionClass.__name__}"
    return QKNormAttention


def llama_qk_norm_nope_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional = None,
    cache_position: Optional[torch.LongTensor] = None,
    type: str = "qk",
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = (
        self.q_norm(self.q_proj(hidden_states))
        if "q" in type
        else self.q_proj(hidden_states)
    )
    key_states = (
        self.k_norm(self.k_proj(hidden_states))
        if "k" in type
        else self.k_proj(hidden_states)
    )
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Null positional embeddings
    original_cos, original_sin = position_embeddings
    cos, sin = (
        torch.ones_like(original_cos),
        torch.zeros_like(original_sin),
    )
    query_states, key_states = llama_apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = llama_eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=(0.0 if not self.training else self.attention_dropout),
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def llama_q_norm_nope_attn_forward(self, *args, **kwargs):
    return llama_qk_norm_nope_attn_forward(self, *args, type="q", **kwargs)


def llama_k_norm_nope_attn_forward(self, *args, **kwargs):
    return llama_qk_norm_nope_attn_forward(self, *args, type="k", **kwargs)


def qwen2_qk_norm_nope_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Any] = None,
    cache_position: Optional[torch.LongTensor] = None,
    type: str = "qk",
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = (
        self.q_norm(self.q_proj(hidden_states))
        if "q" in type
        else self.q_proj(hidden_states)
    )
    key_states = (
        self.k_norm(self.k_proj(hidden_states))
        if "k" in type
        else self.k_proj(hidden_states)
    )
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Null positional embeddings
    original_cos, original_sin = position_embeddings
    cos, sin = (
        torch.ones_like(original_cos),
        torch.zeros_like(original_sin),
    )
    query_states, key_states = qwen2_apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface: Callable = qwen2_eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen2_q_norm_nope_attn_forward(self, *args, **kwargs):
    return qwen2_qk_norm_nope_attn_forward(self, *args, type="q", **kwargs)


def qwen2_k_norm_nope_attn_forward(self, *args, **kwargs):
    return qwen2_qk_norm_nope_attn_forward(self, *args, type="k", **kwargs)


NoPELlamaAttention = nope(LlamaAttention)
NoPEQwen2Attention = nope(Qwen2Attention)

QKNormNoPELlamaAttention = qk_norm(LlamaAttention, llama_qk_norm_nope_attn_forward)
QNormNoPELlamaAttention = qk_norm(LlamaAttention, llama_q_norm_nope_attn_forward)
KNormNoPELlamaAttention = qk_norm(LlamaAttention, llama_k_norm_nope_attn_forward)

QKNormNoPEQwen2Attention = qk_norm(Qwen2Attention, qwen2_qk_norm_nope_attn_forward)
QNormNoPEQwen2Attention = qk_norm(Qwen2Attention, qwen2_q_norm_nope_attn_forward)
KNormNoPEQwen2Attention = qk_norm(Qwen2Attention, qwen2_k_norm_nope_attn_forward)
