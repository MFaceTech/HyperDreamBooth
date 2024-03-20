#!/usr/bin/env python
# coding=utf-8

# Modified by KohakuBlueLeaf
# Modified from diffusers/example/dreambooth/train_dreambooth_lora.py
# see original licensed below
# =======================================================================
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
# =======================================================================

import argparse
import os
from packaging import version
from PIL import Image
import torch
from torchvision.transforms import CenterCrop, Resize, ToTensor, Compose, Normalize
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)


from modules.light_lora import LoRALinearLayer, LoraLoaderMixin
from modules.utils.lora_utils import unet_lora_state_dict, text_encoder_lora_state_dict
from modules.hypernet import HyperDream


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--hypernet_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained hyperkohaku model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        required=True,
        help="Path to reference image",
    )
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="vit_base_patch16_224",
        help="The ViT encoder used in hypernet encoder.",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--down_dim",
        type=int,
        default=160,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--up_dim",
        type=int,
        default=80,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--patch_mlp",
        action="store_true",
        help="Whether to train the text encoder with mlp. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    # Load Model
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float32)
    # pipe.to("cuda")

    unet = pipe.unet
    text_encoder = pipe.text_encoder

    unet_lora_parameters = []
    unet_lora_linear_layers = []
    for i, (attn_processor_name, attn_processor) in enumerate(unet.attn_processors.items()):
        print("unet.attn_processor->%d:%s" % (i, attn_processor_name), attn_processor)
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        print("attn_module:", attn_module)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features,
                rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features,
                rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features,
                rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features, out_features=attn_module.to_out[0].out_features,
                rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
            )
        )
        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        # Accumulate the LoRALinerLayer to optimize.
        unet_lora_linear_layers.append(attn_module.to_q.lora_layer)
        unet_lora_linear_layers.append(attn_module.to_k.lora_layer)
        unet_lora_linear_layers.append(attn_module.to_v.lora_layer)
        unet_lora_linear_layers.append(attn_module.to_out[0].lora_layer)

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_k_proj.in_features,
                    out_features=attn_module.add_k_proj.out_features,
                    rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
                )
            )
            attn_module.add_v_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_v_proj.in_features,
                    out_features=attn_module.add_v_proj.out_features,
                    rank=args.rank, down_dim=args.down_dim, up_dim=args.up_dim, is_train=False,
                )
            )
            unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

            unet_lora_linear_layers.append(attn_module.add_k_proj.lora_layer)
            unet_lora_linear_layers.append(attn_module.add_v_proj.lora_layer)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        # if patch_mlp is True, the finetuning will cover the text encoder mlp,
        # otherwise only the text encoder attention, total lora is (12+12)*4=96
        text_lora_parameters, text_lora_linear_layers = LoraLoaderMixin._modify_text_encoder(text_encoder,
                                                                                             dtype=torch.float32,
                                                                                             rank=args.rank,
                                                                                             down_dim=args.down_dim,
                                                                                             up_dim=args.up_dim,
                                                                                             patch_mlp=args.patch_mlp,
                                                                                             is_train=False)
    # total loras
    lora_linear_layers = unet_lora_linear_layers + text_lora_linear_layers \
        if args.train_text_encoder else unet_lora_linear_layers

    if args.vit_model_name == "vit_base_patch16_224":
        img_encoder_model_name = "vit_base_patch16_224"
        ref_img_size = 224
        mean = [0.5000]
        std = [0.5000]
    elif args.vit_model_name == "vit_huge_patch14_clip_224":
        img_encoder_model_name = "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k"
        ref_img_size = 224
        mean = [0.4815, 0.4578, 0.4082]
        std = [0.2686, 0.2613, 0.2758]
    elif args.vit_model_name == "vit_huge_patch14_clip_336":
        img_encoder_model_name = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
        ref_img_size = 336
        mean = [0.4815, 0.4578, 0.4082]
        std = [0.2686, 0.2613, 0.2758]
    else:
        raise ValueError("%s does not supports!" % args.img_encoder_model_name)

    hypernet_transposes = Compose([
        Resize(size=ref_img_size),
        CenterCrop(size=(ref_img_size, ref_img_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    hypernetwork = HyperDream(
        img_encoder_model_name=img_encoder_model_name,
        ref_img_size=ref_img_size,
        weight_num=len(lora_linear_layers),
        weight_dim=(args.up_dim + args.down_dim) * args.rank,
    )
    hypernetwork.set_lilora(lora_linear_layers)
    
    if os.path.isdir(args.hypernet_model_path):
        path = os.path.join(args.hypernet_model_path, "hypernetwork.bin")
        weight = torch.load(path)
        sd = weight['hypernetwork']
        hypernetwork.load_state_dict(sd)
    else:
        weight = torch.load(args.hypernet_model_path['hypernetwork'])
        sd = weight['hypernetwork']
        hypernetwork.load_state_dict(sd)

    for i, lilora in enumerate(lora_linear_layers):
        seed = weight['aux_seed_%d' % i]
        down_aux = weight['down_aux_%d' % i]
        up_aux = weight['up_aux_%d' % i]
        lilora.update_aux(seed, down_aux, up_aux)

    print(f"Hypernet weights loaded from: {args.hypernet_model_path}")

    hypernetwork = hypernetwork.to("cuda")
    hypernetwork = hypernetwork.eval()

    ref_img = Image.open(args.reference_image_path).convert("RGB")
    ref_img = hypernet_transposes(ref_img).unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        weight, weight_list = hypernetwork(ref_img)
        print("weight>>>>>>>>>>>:",weight.shape, weight)

    # convert down and up weights to linear layer as LoRALinearLayer
    for weight, lora_layer in zip(weight_list, lora_linear_layers):
        lora_layer.update_weight(weight)
        lora_layer.convert_to_standard_lora()

    unet_lora_layers_to_save = unet_lora_state_dict(unet)
    text_encoder_lora_layers_to_save = None
    if args.train_text_encoder:
        text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder, patch_mlp=args.patch_mlp)

    LoraLoaderMixin.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_lora_layers_to_save,
        text_encoder_lora_layers=text_encoder_lora_layers_to_save,
    )

    print("Export LoRA to: %s"%args.output_dir)
    print("==================================complete======================================")


if __name__ == "__main__":
    args = parse_args()
    main(args)