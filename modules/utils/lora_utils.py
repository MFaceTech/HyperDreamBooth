import torch
from typing import Any, Dict, Iterable, Optional, Union
from diffusers.models import UNet2DConditionModel

def unet_lora_state_dict(unet: UNet2DConditionModel) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                current_lora_layer_sd = lora_layer.state_dict()
                for lora_layer_matrix_name, lora_param in current_lora_layer_sd.items():
                    # The matrix name can either be "down" or "up".
                    lora_state_dict[f"{name}.lora.{lora_layer_matrix_name}"] = lora_param

    return lora_state_dict


def text_encoder_lora_state_dict(text_encoder, patch_mlp=False):
    state_dict = {}

    def text_encoder_attn_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        attn_modules = []

        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))

        return attn_modules

    def text_encoder_mlp_modules(text_encoder):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        mlp_modules = []
        if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
            for i, layer in enumerate(text_encoder.text_model.encoder.layers):
                name = f"text_model.encoder.layers.{i}.mlp"
                mod = layer.mlp
                mlp_modules.append((name, mod))

        return mlp_modules

    # text encoder attn layer
    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    # text encoder mlp layer
    if patch_mlp:
        for name, module in text_encoder_mlp_modules(text_encoder):
            for k, v in module.fc1.lora_linear_layer.state_dict().items():
                state_dict[f"{name}.fc1.lora_linear_layer.{k}"] = v

            for k, v in module.fc2.lora_linear_layer.state_dict().items():
                state_dict[f"{name}.fc2.lora_linear_layer.{k}"] = v

    return state_dict
