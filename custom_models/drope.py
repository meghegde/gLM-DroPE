from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    LlamaConfig,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Attention,
    Qwen2Config,
)
from custom_models.attention import (
    NoPELlamaAttention,
    NoPEQwen2Attention,
    QKNormNoPELlamaAttention,
    QNormNoPELlamaAttention,
    KNormNoPELlamaAttention,
    QKNormNoPEQwen2Attention,
    QNormNoPEQwen2Attention,
    KNormNoPEQwen2Attention,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ARCH_MAP = {
    LlamaForCausalLM: (LlamaAttention, LlamaConfig),
    Qwen2ForCausalLM: (Qwen2Attention, Qwen2Config),
}

# Map of attention variants for each base attention class
ATTENTION_VARIANTS = {
    LlamaAttention: {
        "nope": NoPELlamaAttention,
        "qk_norm_nope": QKNormNoPELlamaAttention,
        "q_norm_nope": QNormNoPELlamaAttention,
        "k_norm_nope": KNormNoPELlamaAttention,
    },
    Qwen2Attention: {
        "nope": NoPEQwen2Attention,
        "qk_norm_nope": QKNormNoPEQwen2Attention,
        "q_norm_nope": QNormNoPEQwen2Attention,
        "k_norm_nope": KNormNoPEQwen2Attention,
    },
}


def _create_drope_model_class(
    BaseModelClass,
    **custom_config_fields,
):
    """
    Factory function to create a 'NoPE' version of a Hugging Face model class.

    The returned class inherits from the base model but automatically replaces
    attention layers with custom variants based on config.attention_type.

    Args:
        BaseModelClass: The base model class (e.g., LlamaForCausalLM, Qwen2ForCausalLM)
        **custom_config_fields: Additional config fields with default values.
                               If 'attention_type' is not specified, defaults to 'nope'.

    Available attention types:
        - 'nope': No positional encoding
        - 'qk_norm_nope': QK normalization with no positional encoding
        - 'q_norm_nope': Q-only normalization with no positional encoding
        - 'k_norm_nope': K-only normalization with no positional encoding

    Example:
        # Create model with default 'nope' attention
        model = LlamaDroPE.from_pretrained("meta-llama/Llama-3.2-1B")

        # Create model with K-normalization
        config = LlamaDroPEConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.attention_type = "k_norm_nope"
        model = LlamaDroPE(config)
    """
    if BaseModelClass not in MODEL_ARCH_MAP:
        raise ValueError(
            f"Unknown model architecture: {BaseModelClass}, available architectures: {MODEL_ARCH_MAP.keys()}"
        )

    BaseAttentionClass, BaseConfigClass = MODEL_ARCH_MAP[BaseModelClass]

    # Add default attention_type config field if not specified
    if "attention_type" not in custom_config_fields:
        custom_config_fields["attention_type"] = "nope"

    DroPEConfigClass = _create_drope_config_class(
        BaseConfigClass, **custom_config_fields
    )

    class DroPEModel(BaseModelClass):
        config_class = DroPEConfigClass

        def __init__(self, config, *args, **kwargs):
            # Initialize the base model as usual
            super().__init__(config, *args, **kwargs)
            # After initialization, patch the attention layers
            self._patch_attention_layers()

        def _patch_attention_layers(self):
            """
            Finds and replaces all attention modules with their custom equivalents
            based on the config.attention_type setting.
            """
            # Get the attention type from config
            attention_type = getattr(self.config, "attention_type", "nope")

            # Get the appropriate attention class based on the base attention class and type
            if BaseAttentionClass not in ATTENTION_VARIANTS:
                raise ValueError(
                    f"No attention variants defined for {BaseAttentionClass.__name__}"
                )

            if attention_type not in ATTENTION_VARIANTS[BaseAttentionClass]:
                available_types = list(ATTENTION_VARIANTS[BaseAttentionClass].keys())
                raise ValueError(
                    f"Unknown attention_type '{attention_type}' for {BaseAttentionClass.__name__}. "
                    f"Available types: {available_types}"
                )

            AttentionClass = ATTENTION_VARIANTS[BaseAttentionClass][attention_type]

            # The actual model architecture is often in a `.model` attribute
            model_core = getattr(self, self.base_model_prefix)

            for i, layer in enumerate(model_core.layers):
                # The attention module is usually named 'self_attn'
                if hasattr(layer, "self_attn"):
                    original_attn = layer.self_attn
                    # Ensure we are only patching the intended class
                    if isinstance(original_attn, BaseAttentionClass):
                        new_attn = AttentionClass.from_source(original_attn, self.config)
                        layer.self_attn = new_attn
                        logger.debug(
                            f"Replaced attention in layer {i} with {new_attn.__class__.__name__}"
                        )
                else:
                    logger.warning(
                        f"Could not find 'self_attn' in layer {i} of {self.__class__.__name__}"
                    )

    # Register the new class with the AutoModel framework for HF compatibility.
    # This is important for .from_pretrained() and .save_pretrained() to work correctly.
    architecture_name = f"DroPE{BaseModelClass.__name__}"

    # Give our new class a descriptive name
    DroPEModel.__name__ = architecture_name

    # Register model class with the AutoModel framework for HF compatibility.
    AutoModelForCausalLM.register(DroPEModel.config_class, DroPEModel)

    # Register with auto_map so it can be loaded by remote code
    DroPEModel.register_for_auto_class("AutoModelForCausalLM")

    return DroPEModel


def _create_drope_config_class(
    BaseConfigClass: type[PretrainedConfig], **custom_fields
):
    """
    Factory function to create a custom config class with additional fields.

    Args:
        BaseConfigClass: The base config class to inherit from (e.g., LlamaConfig).
        **custom_fields: Keyword arguments defining the new config fields and their
                         default values (e.g., use_qk_norm=False).

    Returns:
        A new custom config class.
    """

    class DroPEConfig(BaseConfigClass):
        # Override model_type to match the custom model type
        model_type = f"{BaseConfigClass.model_type}_drope"

        # By adding the new fields to the class attributes, they are recognized
        # by the Hugging Face `from_pretrained` mechanism.
        def __init__(self, **kwargs):
            # Pop our custom fields from kwargs, using the defaults from the factory
            for field, default_value in custom_fields.items():
                setattr(self, field, kwargs.pop(field, default_value))

            # Initialize the parent class with the remaining kwargs
            super().__init__(**kwargs)

        @classmethod
        def from_base_config(cls, base_config: BaseConfigClass):
            return DroPEConfig(
                **base_config.to_dict(),
            )

    # Give the new class a descriptive name for clarity
    DroPEConfig.__name__ = f"DroPE{BaseConfigClass.__name__}"

    # Register the new config class with the AutoConfig framework for HF compatibility.
    AutoConfig.register(DroPEConfig.model_type, DroPEConfig)

    # Register with auto_map so it can be loaded by remote code
    DroPEConfig.register_for_auto_class()

    return DroPEConfig


DroPEQwen2Config = _create_drope_config_class(Qwen2Config, attention_type="nope")
DroPELlamaConfig = _create_drope_config_class(LlamaConfig, attention_type="nope")

DroPEQwen2ForCausalLM = _create_drope_model_class(Qwen2ForCausalLM)
DroPELlamaForCausalLM = _create_drope_model_class(LlamaForCausalLM)