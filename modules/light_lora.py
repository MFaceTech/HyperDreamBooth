# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTANT:                                                      #
###################################################################
# ----------------------------------------------------------------#
# This file is deprecated and will be removed soon                #
# (as soon as PEFT will become a required dependency for LoRA)    #
# ----------------------------------------------------------------#
###################################################################

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.utils.import_utils import is_transformers_available


if is_transformers_available():
    from transformers import CLIPTextModel, CLIPTextModelWithProjection


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f"text_model.encoder.layers.{i}.mlp"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale
            attn_module.k_proj.lora_scale = lora_scale
            attn_module.v_proj.lora_scale = lora_scale
            attn_module.out_proj.lora_scale = lora_scale

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        if isinstance(mlp_module.fc1, PatchedLoraProjection):
            mlp_module.fc1.lora_scale = lora_scale
            mlp_module.fc2.lora_scale = lora_scale


class PatchedLoraProjection(torch.nn.Module):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, down_dim=160, up_dim=80, dtype=None, is_train=True):
        super().__init__()

        self.regular_linear_layer = regular_linear_layer

        device = self.regular_linear_layer.weight.device

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
            rank=rank,
            down_dim=down_dim,
            up_dim=up_dim,
            is_train=is_train,
        )

        self.lora_scale = lora_scale

    # overwrite PyTorch's `state_dict` to be sure that only the 'regular_linear_layer' weights are saved
    # when saving the whole text encoder model and when LoRA is unloaded or fused
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if self.lora_linear_layer is None:
            return self.regular_linear_layer.state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    def _fuse_lora(self, lora_scale=1.0, safe_fusing=False):
        if self.lora_linear_layer is None:
            return

        dtype, device = self.regular_linear_layer.weight.data.dtype, self.regular_linear_layer.weight.data.device

        w_orig = self.regular_linear_layer.weight.data.float()
        w_up = self.lora_linear_layer.up.weight.data.float()
        w_down = self.lora_linear_layer.down.weight.data.float()

        if self.lora_linear_layer.network_alpha is not None:
            w_up = w_up * self.lora_linear_layer.network_alpha / self.lora_linear_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.regular_linear_layer.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_linear_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self.lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.regular_linear_layer.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self.lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.regular_linear_layer.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, input):
        if self.lora_scale is None:
            self.lora_scale = 1.0
        if self.lora_linear_layer is None:
            return self.regular_linear_layer(input)
        return self.regular_linear_layer(input) + (self.lora_scale * self.lora_linear_layer(input))

class LoRALinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 4,
            down_dim: int = 160,
            up_dim: int = 80,
            network_alpha: Optional[float] = None,
            is_train: bool = False,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
        # register_buffer is not a learnable parameter, but is a part of the module’s state
        self.register_buffer('aux_seed', torch.randint(0, 2 ** 32 - 1, (1,)))

        # down_aux and up_aux: orthonogal initialization
        torch.manual_seed(self.aux_seed.item())
        down_aux = torch.empty(down_dim, in_features)
        up_aux = torch.empty(out_features, up_dim)
        torch.nn.init.orthogonal_(down_aux, gain=1)
        torch.nn.init.orthogonal_(up_aux, gain=1)

        # register_buffer is not a learnable parameter, but is a part of the module’s state
        self.register_buffer('down_aux', down_aux)
        self.register_buffer('up_aux', up_aux)

        self.in_features = in_features
        self.out_features = out_features
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.down = self.up = None
        self.network_alpha = network_alpha
        self.rank = rank
        self.is_train = is_train

        if is_train:
            # weight initialization
            down_weight = torch.empty(rank, down_dim)
            up_weight = torch.empty(up_dim, rank)
            nn.init.xavier_normal_(down_weight)
            nn.init.zeros_(up_weight)
            weight = torch.concat([torch.flatten(down_weight), torch.flatten(up_weight)])
            self.weight_embedding = nn.Parameter(weight)

    def update_weight(self, weight_embedding):
        """
           Update the weights of the LoRa model. Only called outside.
           This function is used to replace the weights of the LoRA model with pre-optimized weights or weights predicted
           by a hypernetwork.
           """
        # Check if the shape is [batch_size, (up_dim+down_dim)*r] or [(up_dim+down_dim)*r]
        expected_dim = (self.up_dim + self.down_dim) * self.rank

        if len(weight_embedding.shape) > 2 or (
                len(weight_embedding.shape) == 2 and weight_embedding.shape[1] != expected_dim) or (
                len(weight_embedding.shape) == 1 and weight_embedding.shape != (expected_dim,)):
            raise ValueError(
                "The shape of weight_embedding must be [batch_size, (up_dim+down_dim)*r] or [(up_dim+down_dim)*r], "
                "and the dimension of weight_embedding must not be more than 2.")
        self.weight_embedding = weight_embedding

    def update_aux(self, aux_seed=None, down_aux=None, up_aux=None):
        """
        update up_aux and down_aux based on aux_seed, using orthonogal initialization
        """
        self.aux_seed = aux_seed
        self.down_aux = down_aux
        self.up_aux = up_aux


    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.weight_embedding.dtype

        # get weight
        down_aux = self.down_aux.to(self.weight_embedding.device)
        up_aux = self.up_aux.to(self.weight_embedding.device)
        down_weight, up_weight = self.weight_embedding.split([self.down_dim * self.rank, self.up_dim * self.rank], dim=-1)
        # reshape weight
        if self.weight_embedding.dim() == 1:
            # for local lora training
            down_weight = down_weight.reshape(self.rank, -1)
            up_weight = up_weight.reshape(-1, self.rank)
        elif self.weight_embedding.dim() == 2:
            # for PreOptHyperNet and HyperNet training, support batch training
            down_weight = down_weight.reshape(self.weight_embedding.size(0), self.rank, -1)
            up_weight = up_weight.reshape(self.weight_embedding.size(0), -1, self.rank)
        else:
            raise ValueError(f"weight_embedding dim {self.weight_embedding.dim()} is not supported")

        # TODO: make weight, matrix multiplication
        down = down_weight @ down_aux  # [batch, rank, in] or [rank, in]
        up = up_aux @ up_weight  # [batch, out, rank] or [out, rank]

        # linear layer
        if down.dim() == 3:
            # Each sample has its own weight matrix if it is 3D
            down_hidden_states = torch.einsum('b o i, b ... i -> b ... o', down, hidden_states.to(dtype))
            up_hidden_states = torch.einsum('b o i, b ... i -> b ... o', up, down_hidden_states)
        else:
            # All samples share the same weight matrix if it is 2D
            down_hidden_states = F.linear(hidden_states.to(dtype), down)
            up_hidden_states = F.linear(down_hidden_states, up)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

    def convert_to_standard_lora(self):
        # weight_embedding shape : [(down_dim+up_dim)*rank] or [1, (down_dim+up_dim)*rank]
        assert self.weight_embedding is not None and self.down_aux is not None and self.up_aux is not None
        assert (self.weight_embedding.dim() == 1 or (self.weight_embedding.dim() == 2 and self.weight_embedding.size(0) == 1))
        # get weight
        down_aux = self.down_aux.to(self.weight_embedding.device)
        up_aux = self.up_aux.to(self.weight_embedding.device)
        # reshape weight
        down_weight, up_weight = self.weight_embedding.split([self.down_dim * self.rank, self.up_dim * self.rank], dim=-1)
        down_weight = down_weight.reshape(self.rank, -1)
        up_weight = up_weight.reshape(-1, self.rank)
        # TODO: make weight, matrix multiplication
        down = down_weight @ down_aux
        up = up_aux @ up_weight

        # convert to linear weight
        with torch.no_grad():
            self.down = nn.Linear(self.in_features, self.rank, bias=False)
            self.down.weight = nn.Parameter(down)
            self.up = nn.Linear(self.rank, self.out_features, bias=False)
            self.up.weight = nn.Parameter(up)

        # remove reduntant variables
        del self.aux_seed, self.down_aux, self.up_aux
        del self.weight_embedding


class LoRAConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((w_orig.shape))
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return original_outputs + (scale * self.lora_layer(hidden_states))


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out




# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
from huggingface_hub import model_info
from packaging import version
from torch import nn

from diffusers import __version__
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta
from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    USE_PEFT_BACKEND,
    _get_model_file,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    delete_adapter_layers,
    deprecate,
    get_adapter_name,
    get_peft_kwargs,
    is_accelerate_available,
    is_transformers_available,
    logging,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from diffusers.loaders.lora_conversion_utils import _convert_kohya_lora_to_diffusers, _maybe_map_sgm_blocks_to_diffusers


if is_transformers_available():
    from transformers import PreTrainedModel

    # from lora import PatchedLoraProjection, text_encoder_attn_modules, text_encoder_mlp_modules

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

LORA_DEPRECATION_MESSAGE = "You are using an old version of LoRA backend. This will be deprecated in the next releases in favor of PEFT make sure to install the latest PEFT and transformers packages in the future."


class LoraLoaderMixin:
    r"""
    Load LoRA layers into [`UNet2DConditionModel`] and
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME
    num_fused_loras = 0

    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], adapter_name=None, **kwargs
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
        )
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=getattr(self, self.text_encoder_name)
            if not hasattr(self, "text_encoder")
            else self.text_encoder,
            lora_scale=self.lora_scale,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        unet_config = kwargs.pop("unet_config", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    # Here we're relaxing the loading check to enable more Inference API
                    # friendliness where sometimes, it's not at all possible to automatically
                    # determine `weight_name`.
                    if weight_name is None:
                        weight_name = cls._best_guess_weight_name(
                            pretrained_model_name_or_path_or_dict, file_extension=".safetensors"
                        )
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except (IOError, safetensors.SafetensorError) as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    model_file = None
                    pass

            if model_file is None:
                if weight_name is None:
                    weight_name = cls._best_guess_weight_name(
                        pretrained_model_name_or_path_or_dict, file_extension=".bin"
                    )
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        network_alphas = None
        # TODO: replace it with a method from `state_dict_utils`
        if all(
            (
                k.startswith("lora_te_")
                or k.startswith("lora_unet_")
                or k.startswith("lora_te1_")
                or k.startswith("lora_te2_")
            )
            for k in state_dict.keys()
        ):
            # Map SDXL blocks correctly.
            if unet_config is not None:
                # use unet config to remap block numbers
                state_dict = _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)
            state_dict, network_alphas = _convert_kohya_lora_to_diffusers(state_dict)

        return state_dict, network_alphas

    @classmethod
    def _best_guess_weight_name(cls, pretrained_model_name_or_path_or_dict, file_extension=".safetensors"):
        targeted_files = []

        if os.path.isfile(pretrained_model_name_or_path_or_dict):
            return
        elif os.path.isdir(pretrained_model_name_or_path_or_dict):
            targeted_files = [
                f for f in os.listdir(pretrained_model_name_or_path_or_dict) if f.endswith(file_extension)
            ]
        else:
            files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
            targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]
        if len(targeted_files) == 0:
            return

        # "scheduler" does not correspond to a LoRA checkpoint.
        # "optimizer" does not correspond to a LoRA checkpoint
        # only top-level checkpoints are considered and not the other ones, hence "checkpoint".
        unallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
        targeted_files = list(
            filter(lambda x: all(substring not in x for substring in unallowed_substrings), targeted_files)
        )

        if any(f.endswith(LORA_WEIGHT_NAME) for f in targeted_files):
            targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME), targeted_files))
        elif any(f.endswith(LORA_WEIGHT_NAME_SAFE) for f in targeted_files):
            targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME_SAFE), targeted_files))

        if len(targeted_files) > 1:
            raise ValueError(
                f"Provided path contains more than one weights file in the {file_extension} format. Either specify `weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  {pretrained_model_name_or_path_or_dict}."
            )
        weight_name = targeted_files[0]
        return weight_name

    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        if _pipeline is not None:
            for _, component in _pipeline.components.items():
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    if not is_model_cpu_offload:
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    if not is_sequential_cpu_offload:
                        is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        return (is_model_cpu_offload, is_sequential_cpu_offload)

    @classmethod
    def load_lora_into_unet(
        cls, state_dict, network_alphas, unet, low_cpu_mem_usage=None, adapter_name=None, _pipeline=None
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `unet`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        low_cpu_mem_usage = low_cpu_mem_usage if low_cpu_mem_usage is not None else _LOW_CPU_MEM_USAGE_DEFAULT
        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `cls.unet_name` and/or `cls.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())

        if all(key.startswith("unet.unet") for key in keys):
            deprecation_message = "Keys starting with 'unet.unet' are deprecated."
            deprecate("unet.unet keys", "0.27", deprecation_message)

        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to UNet.
            logger.info(f"Loading {cls.unet_name}.")

            unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
            state_dict = {k.replace(f"{cls.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

            if network_alphas is not None:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(cls.unet_name)]
                network_alphas = {
                    k.replace(f"{cls.unet_name}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                }

        else:
            # Otherwise, we're dealing with the old format. This means the `state_dict` should only
            # contain the module names of the `unet` as its keys WITHOUT any prefix.
            if not USE_PEFT_BACKEND:
                warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
                logger.warn(warn_message)

        if USE_PEFT_BACKEND and len(state_dict.keys()) > 0:
            from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

            if adapter_name in getattr(unet, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the Unet - please select a new adapter name."
                )

            state_dict = convert_unet_state_dict_to_peft(state_dict)

            if network_alphas is not None:
                # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                # `convert_unet_state_dict_to_peft` method.
                network_alphas = convert_unet_state_dict_to_peft(network_alphas)

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(unet)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            is_model_cpu_offload, is_sequential_cpu_offload = cls._optionally_disable_offloading(_pipeline)

            inject_adapter_in_model(lora_config, unet, adapter_name=adapter_name)
            incompatible_keys = set_peft_model_state_dict(unet, state_dict, adapter_name)

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />

        unet.load_attn_procs(
            state_dict, network_alphas=network_alphas, low_cpu_mem_usage=low_cpu_mem_usage, _pipeline=_pipeline
        )

    @classmethod
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        low_cpu_mem_usage=None,
        adapter_name=None,
        _pipeline=None,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `text_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key should be prefixed with an
                additional `text_encoder` to distinguish between unet lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            text_encoder (`CLIPTextModel`):
                The text encoder model to load the LoRA layers into.
            prefix (`str`):
                Expected prefix of the `text_encoder` in the `state_dict`.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        low_cpu_mem_usage = low_cpu_mem_usage if low_cpu_mem_usage is not None else _LOW_CPU_MEM_USAGE_DEFAULT

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        prefix = cls.text_encoder_name if prefix is None else prefix

        # Safe prefix to check with.
        if any(cls.text_encoder_name in key for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
            text_encoder_lora_state_dict = {
                k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }

            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {prefix}.")
                rank = {}
                text_encoder_lora_state_dict = convert_state_dict_to_diffusers(text_encoder_lora_state_dict)

                if USE_PEFT_BACKEND:
                    # convert state dict
                    text_encoder_lora_state_dict = convert_state_dict_to_peft(text_encoder_lora_state_dict)

                    for name, _ in text_encoder_attn_modules(text_encoder):
                        rank_key = f"{name}.out_proj.lora_B.weight"
                        rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

                    patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                    if patch_mlp:
                        for name, _ in text_encoder_mlp_modules(text_encoder):
                            rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                            rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                            rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                            rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]
                else:
                    for name, _ in text_encoder_attn_modules(text_encoder):
                        rank_key = f"{name}.out_proj.lora_linear_layer.up.weight"
                        rank.update({rank_key: text_encoder_lora_state_dict[rank_key].shape[1]})

                    patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                    if patch_mlp:
                        for name, _ in text_encoder_mlp_modules(text_encoder):
                            rank_key_fc1 = f"{name}.fc1.lora_linear_layer.up.weight"
                            rank_key_fc2 = f"{name}.fc2.lora_linear_layer.up.weight"
                            rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[1]
                            rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[1]

                if network_alphas is not None:
                    alpha_keys = [
                        k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix
                    ]
                    network_alphas = {
                        k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                if USE_PEFT_BACKEND:
                    from peft import LoraConfig

                    lora_config_kwargs = get_peft_kwargs(
                        rank, network_alphas, text_encoder_lora_state_dict, is_unet=False
                    )

                    lora_config = LoraConfig(**lora_config_kwargs)

                    # adapter_name
                    if adapter_name is None:
                        adapter_name = get_adapter_name(text_encoder)

                    is_model_cpu_offload, is_sequential_cpu_offload = cls._optionally_disable_offloading(_pipeline)

                    # inject LoRA layers and load the state dict
                    # in transformers we automatically check whether the adapter name is already in use or not
                    text_encoder.load_adapter(
                        adapter_name=adapter_name,
                        adapter_state_dict=text_encoder_lora_state_dict,
                        peft_config=lora_config,
                    )

                    # scale LoRA layers with `lora_scale`
                    scale_lora_layers(text_encoder, weight=lora_scale)
                else:
                    cls._modify_text_encoder(
                        text_encoder,
                        lora_scale,
                        network_alphas,
                        rank=rank,
                        patch_mlp=patch_mlp,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )

                    is_pipeline_offloaded = _pipeline is not None and any(
                        isinstance(c, torch.nn.Module) and hasattr(c, "_hf_hook")
                        for c in _pipeline.components.values()
                    )
                    if is_pipeline_offloaded and low_cpu_mem_usage:
                        low_cpu_mem_usage = True
                        logger.info(
                            f"Pipeline {_pipeline.__class__} is offloaded. Therefore low cpu mem usage loading is forced."
                        )

                    if low_cpu_mem_usage:
                        device = next(iter(text_encoder_lora_state_dict.values())).device
                        dtype = next(iter(text_encoder_lora_state_dict.values())).dtype
                        unexpected_keys = load_model_dict_into_meta(
                            text_encoder, text_encoder_lora_state_dict, device=device, dtype=dtype
                        )
                    else:
                        load_state_dict_results = text_encoder.load_state_dict(
                            text_encoder_lora_state_dict, strict=False
                        )
                        unexpected_keys = load_state_dict_results.unexpected_keys

                    if len(unexpected_keys) != 0:
                        raise ValueError(
                            f"failed to load text encoder state dict, unexpected keys: {load_state_dict_results.unexpected_keys}"
                        )

                    # <Unsafe code
                    # We can be sure that the following works as all we do is change the dtype and device of the text encoder
                    # Now we remove any existing hooks to
                    is_model_cpu_offload = False
                    is_sequential_cpu_offload = False
                    if _pipeline is not None:
                        for _, component in _pipeline.components.items():
                            if isinstance(component, torch.nn.Module):
                                if hasattr(component, "_hf_hook"):
                                    is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                                    is_sequential_cpu_offload = isinstance(
                                        getattr(component, "_hf_hook"), AlignDevicesHook
                                    )
                                    logger.info(
                                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                                    )
                                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

                text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

                # Offload back.
                if is_model_cpu_offload:
                    _pipeline.enable_model_cpu_offload()
                elif is_sequential_cpu_offload:
                    _pipeline.enable_sequential_cpu_offload()
                # Unsafe code />

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    def _remove_text_encoder_monkey_patch(self):
        if USE_PEFT_BACKEND:
            remove_method = recurse_remove_peft_layers
        else:
            remove_method = self._remove_text_encoder_monkey_patch_classmethod

        if hasattr(self, "text_encoder"):
            remove_method(self.text_encoder)

            # In case text encoder have no Lora attached
            if USE_PEFT_BACKEND and getattr(self.text_encoder, "peft_config", None) is not None:
                del self.text_encoder.peft_config
                self.text_encoder._hf_peft_config_loaded = None
        if hasattr(self, "text_encoder_2"):
            remove_method(self.text_encoder_2)
            if USE_PEFT_BACKEND:
                del self.text_encoder_2.peft_config
                self.text_encoder_2._hf_peft_config_loaded = None

    @classmethod
    def _remove_text_encoder_monkey_patch_classmethod(cls, text_encoder):
        deprecate("_remove_text_encoder_monkey_patch_classmethod", "0.27", LORA_DEPRECATION_MESSAGE)

        for _, attn_module in text_encoder_attn_modules(text_encoder):
            if isinstance(attn_module.q_proj, PatchedLoraProjection):
                attn_module.q_proj.lora_linear_layer = None
                attn_module.k_proj.lora_linear_layer = None
                attn_module.v_proj.lora_linear_layer = None
                attn_module.out_proj.lora_linear_layer = None

        for _, mlp_module in text_encoder_mlp_modules(text_encoder):
            if isinstance(mlp_module.fc1, PatchedLoraProjection):
                mlp_module.fc1.lora_linear_layer = None
                mlp_module.fc2.lora_linear_layer = None

    @classmethod
    def _modify_text_encoder(
        cls,
        text_encoder,
        lora_scale=1,
        network_alphas=None,
        rank: Union[Dict[str, int], int] = 4,
        down_dim: Union[Dict[str, int], int] = 160,
        up_dim: Union[Dict[str, int], int] = 80,
        dtype=None,
        patch_mlp=False,
        low_cpu_mem_usage=False,
        is_train=True,
    ):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """
        deprecate("_modify_text_encoder", "0.27", LORA_DEPRECATION_MESSAGE)

        def create_patched_linear_lora(model, network_alpha, rank, down_dim, up_dim, dtype, lora_parameters, lora_linear_layers, is_train):
            linear_layer = model.regular_linear_layer if isinstance(model, PatchedLoraProjection) else model
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                model = PatchedLoraProjection(linear_layer, lora_scale, network_alpha, rank, down_dim, up_dim, dtype=dtype, is_train=is_train)

            lora_parameters.extend(model.lora_linear_layer.parameters())
            # TODO: for User-defined LoRA
            lora_linear_layers.append(model.lora_linear_layer)
            return model

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []
        lora_linear_layers = []
        network_alphas = {} if network_alphas is None else network_alphas
        is_network_alphas_populated = len(network_alphas) > 0

        for i, (name, attn_module) in enumerate(text_encoder_attn_modules(text_encoder)):
            print("text_encoder.attn_module->%d:%s"%(i, name), attn_module)
            query_alpha = network_alphas.pop(name + ".to_q_lora.down.weight.alpha", None)
            key_alpha = network_alphas.pop(name + ".to_k_lora.down.weight.alpha", None)
            value_alpha = network_alphas.pop(name + ".to_v_lora.down.weight.alpha", None)
            out_alpha = network_alphas.pop(name + ".t_out_lora.down.weight.alpha", None)

            if isinstance(rank, dict):
                current_rank = rank.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
            else:
                current_rank = rank
            if isinstance(down_dim, dict):
                current_down_dim = down_dim.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
            else:
                current_down_dim = down_dim
            if isinstance(up_dim, dict):
                current_up_dim = up_dim.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
            else:
                current_up_dim = up_dim

            attn_module.q_proj = create_patched_linear_lora(
                attn_module.q_proj, query_alpha, current_rank, current_down_dim, current_up_dim, dtype, lora_parameters, lora_linear_layers, is_train
            )
            attn_module.k_proj = create_patched_linear_lora(
                attn_module.k_proj, key_alpha, current_rank, current_down_dim, current_up_dim, dtype, lora_parameters, lora_linear_layers, is_train
            )
            attn_module.v_proj = create_patched_linear_lora(
                attn_module.v_proj, value_alpha, current_rank, current_down_dim, current_up_dim, dtype, lora_parameters, lora_linear_layers, is_train
            )
            attn_module.out_proj = create_patched_linear_lora(
                attn_module.out_proj, out_alpha, current_rank, current_down_dim, current_up_dim, dtype, lora_parameters, lora_linear_layers, is_train
            )

        if patch_mlp:
            for i, (name, mlp_module) in enumerate(text_encoder_mlp_modules(text_encoder)):
                print("text_encoder.mlp_module->%d:%s" % (i, name), mlp_module)

                fc1_alpha = network_alphas.pop(name + ".fc1.lora_linear_layer.down.weight.alpha", None)
                fc2_alpha = network_alphas.pop(name + ".fc2.lora_linear_layer.down.weight.alpha", None)
                if isinstance(rank, dict):
                    current_rank_fc1 = rank.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                    current_rank_fc2 = rank.pop(f"{name}.fc2.lora_linear_layer.up.weight")
                else:
                    current_rank_fc1 = rank
                    current_rank_fc2 = rank
                if isinstance(down_dim, dict):
                    current_down_dim_fc1 = down_dim.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                    current_down_dim_fc2 = down_dim.pop(f"{name}.fc2.lora_linear_layer.up.weight")
                else:
                    current_down_dim_fc1 = down_dim
                    current_down_dim_fc2 = down_dim
                if isinstance(up_dim, dict):
                    current_up_dim_fc1 = up_dim.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                    current_up_dim_fc2 = up_dim.pop(f"{name}.fc2.lora_linear_layer.up.weight")
                else:
                    current_up_dim_fc1 = up_dim
                    current_up_dim_fc2 = up_dim


                mlp_module.fc1 = create_patched_linear_lora(
                    mlp_module.fc1, fc1_alpha, current_rank_fc1, current_down_dim_fc1, current_up_dim_fc1, dtype, lora_parameters, lora_linear_layers, is_train
                )
                mlp_module.fc2 = create_patched_linear_lora(
                    mlp_module.fc2, fc2_alpha, current_rank_fc2, current_down_dim_fc2, current_up_dim_fc2, dtype, lora_parameters, lora_linear_layers, is_train
                )

        if is_network_alphas_populated and len(network_alphas) > 0:
            raise ValueError(
                f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
            )

        return lora_parameters, lora_linear_layers

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or text_encoder_lora_layers):
            raise ValueError("You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers`.")

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, torch.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
    ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        if not USE_PEFT_BACKEND:
            if version.parse(__version__) > version.parse("0.23"):
                logger.warn(
                    "You are using `unload_lora_weights` to disable and unload lora weights. If you want to iteratively enable and disable adapter weights,"
                    "you can use `pipe.enable_lora()` or `pipe.disable_lora()`. After installing the latest version of PEFT."
                )

            for _, module in self.unet.named_modules():
                if hasattr(module, "set_lora_layer"):
                    module.set_lora_layer(None)
        else:
            recurse_remove_peft_layers(self.unet)
            if hasattr(self.unet, "peft_config"):
                del self.unet.peft_config

        # Safe to call the following regardless of LoRA.
        self._remove_text_encoder_monkey_patch()

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            fuse_unet (`bool`, defaults to `True`): Whether to fuse the UNet LoRA parameters.
            fuse_text_encoder (`bool`, defaults to `True`):
                Whether to fuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
        """
        if fuse_unet or fuse_text_encoder:
            self.num_fused_loras += 1
            if self.num_fused_loras > 1:
                logger.warn(
                    "The current API is supported for operating with a single LoRA file. You are trying to load and fuse more than one LoRA which is not well-supported.",
                )

        if fuse_unet:
            self.unet.fuse_lora(lora_scale, safe_fusing=safe_fusing)

        if USE_PEFT_BACKEND:
            from peft.tuners.tuners_utils import BaseTunerLayer

            def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False):
                # TODO(Patrick, Younes): enable "safe" fusing
                for module in text_encoder.modules():
                    if isinstance(module, BaseTunerLayer):
                        if lora_scale != 1.0:
                            module.scale_layer(lora_scale)

                        module.merge()

        else:
            deprecate("fuse_text_encoder_lora", "0.27", LORA_DEPRECATION_MESSAGE)

            def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False):
                for _, attn_module in text_encoder_attn_modules(text_encoder):
                    if isinstance(attn_module.q_proj, PatchedLoraProjection):
                        attn_module.q_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.k_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.v_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.out_proj._fuse_lora(lora_scale, safe_fusing)

                for _, mlp_module in text_encoder_mlp_modules(text_encoder):
                    if isinstance(mlp_module.fc1, PatchedLoraProjection):
                        mlp_module.fc1._fuse_lora(lora_scale, safe_fusing)
                        mlp_module.fc2._fuse_lora(lora_scale, safe_fusing)

        if fuse_text_encoder:
            if hasattr(self, "text_encoder"):
                fuse_text_encoder_lora(self.text_encoder, lora_scale, safe_fusing)
            if hasattr(self, "text_encoder_2"):
                fuse_text_encoder_lora(self.text_encoder_2, lora_scale, safe_fusing)

    def unfuse_lora(self, unfuse_unet: bool = True, unfuse_text_encoder: bool = True):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        if unfuse_unet:
            if not USE_PEFT_BACKEND:
                self.unet.unfuse_lora()
            else:
                from peft.tuners.tuners_utils import BaseTunerLayer

                for module in self.unet.modules():
                    if isinstance(module, BaseTunerLayer):
                        module.unmerge()

        if USE_PEFT_BACKEND:
            from peft.tuners.tuners_utils import BaseTunerLayer

            def unfuse_text_encoder_lora(text_encoder):
                for module in text_encoder.modules():
                    if isinstance(module, BaseTunerLayer):
                        module.unmerge()

        else:
            deprecate("unfuse_text_encoder_lora", "0.27", LORA_DEPRECATION_MESSAGE)

            def unfuse_text_encoder_lora(text_encoder):
                for _, attn_module in text_encoder_attn_modules(text_encoder):
                    if isinstance(attn_module.q_proj, PatchedLoraProjection):
                        attn_module.q_proj._unfuse_lora()
                        attn_module.k_proj._unfuse_lora()
                        attn_module.v_proj._unfuse_lora()
                        attn_module.out_proj._unfuse_lora()

                for _, mlp_module in text_encoder_mlp_modules(text_encoder):
                    if isinstance(mlp_module.fc1, PatchedLoraProjection):
                        mlp_module.fc1._unfuse_lora()
                        mlp_module.fc2._unfuse_lora()

        if unfuse_text_encoder:
            if hasattr(self, "text_encoder"):
                unfuse_text_encoder_lora(self.text_encoder)
            if hasattr(self, "text_encoder_2"):
                unfuse_text_encoder_lora(self.text_encoder_2)

        self.num_fused_loras -= 1

    def set_adapters_for_text_encoder(
        self,
        adapter_names: Union[List[str], str],
        text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
        text_encoder_weights: List[float] = None,
    ):
        """
        Sets the adapter layers for the text encoder.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
                attribute.
            text_encoder_weights (`List[float]`, *optional*):
                The weights to use for the text encoder. If `None`, the weights are set to `1.0` for all the adapters.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        def process_weights(adapter_names, weights):
            if weights is None:
                weights = [1.0] * len(adapter_names)
            elif isinstance(weights, float):
                weights = [weights]

            if len(adapter_names) != len(weights):
                raise ValueError(
                    f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}"
                )
            return weights

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        text_encoder_weights = process_weights(adapter_names, text_encoder_weights)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError(
                "The pipeline does not have a default `pipe.text_encoder` class. Please make sure to pass a `text_encoder` instead."
            )
        set_weights_and_activate_adapters(text_encoder, adapter_names, text_encoder_weights)

    def disable_lora_for_text_encoder(self, text_encoder: Optional["PreTrainedModel"] = None):
        """
        Disables the LoRA layers for the text encoder.

        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to disable the LoRA layers for. If `None`, it will try to get the
                `text_encoder` attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(text_encoder, enabled=False)

    def enable_lora_for_text_encoder(self, text_encoder: Optional["PreTrainedModel"] = None):
        """
        Enables the LoRA layers for the text encoder.

        Args:
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to enable the LoRA layers for. If `None`, it will try to get the `text_encoder`
                attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(self.text_encoder, enabled=True)

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[List[float]] = None,
    ):
        # Handle the UNET
        self.unet.set_adapters(adapter_names, adapter_weights)

        # Handle the Text Encoder
        if hasattr(self, "text_encoder"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder, adapter_weights)
        if hasattr(self, "text_encoder_2"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder_2, adapter_weights)

    def disable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Disable unet adapters
        self.unet.disable_lora()

        # Disable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.disable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.disable_lora_for_text_encoder(self.text_encoder_2)

    def enable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Enable unet adapters
        self.unet.enable_lora()

        # Enable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.enable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.enable_lora_for_text_encoder(self.text_encoder_2)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Args:
        Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).
            adapter_names (`Union[List[str], str]`):
                The names of the adapter to delete. Can be a single string or a list of strings
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        self.unet.delete_adapters(adapter_names)

        for adapter_name in adapter_names:
            # Delete text encoder adapters
            if hasattr(self, "text_encoder"):
                delete_adapter_layers(self.text_encoder, adapter_name)
            if hasattr(self, "text_encoder_2"):
                delete_adapter_layers(self.text_encoder_2, adapter_name)

    def get_active_adapters(self) -> List[str]:
        """
        Gets the list of the current active adapters.

        Example:

        ```python
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
        ).to("cuda")
        pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipeline.get_active_adapters()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer

        active_adapters = []

        for module in self.unet.modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapters
                break

        return active_adapters

    def get_list_adapters(self) -> Dict[str, List[str]]:
        """
        Gets the current list of all available adapters in the pipeline.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        set_adapters = {}

        if hasattr(self, "text_encoder") and hasattr(self.text_encoder, "peft_config"):
            set_adapters["text_encoder"] = list(self.text_encoder.peft_config.keys())

        if hasattr(self, "text_encoder_2") and hasattr(self.text_encoder_2, "peft_config"):
            set_adapters["text_encoder_2"] = list(self.text_encoder_2.peft_config.keys())

        if hasattr(self, "unet") and hasattr(self.unet, "peft_config"):
            set_adapters["unet"] = list(self.unet.peft_config.keys())

        return set_adapters

    def set_lora_device(self, adapter_names: List[str], device: Union[torch.device, str, int]) -> None:
        """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[torch.device, str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        # Handle the UNET
        for unet_module in self.unet.modules():
            if isinstance(unet_module, BaseTunerLayer):
                for adapter_name in adapter_names:
                    unet_module.lora_A[adapter_name].to(device)
                    unet_module.lora_B[adapter_name].to(device)

        # Handle the text encoder
        modules_to_process = []
        if hasattr(self, "text_encoder"):
            modules_to_process.append(self.text_encoder)

        if hasattr(self, "text_encoder_2"):
            modules_to_process.append(self.text_encoder_2)

        for text_encoder in modules_to_process:
            # loop over submodules
            for text_encoder_module in text_encoder.modules():
                if isinstance(text_encoder_module, BaseTunerLayer):
                    for adapter_name in adapter_names:
                        text_encoder_module.lora_A[adapter_name].to(device)
                        text_encoder_module.lora_B[adapter_name].to(device)

