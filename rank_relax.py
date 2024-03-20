import os
import time
import itertools
from PIL import Image
from torchvision.transforms import CenterCrop, Resize, ToTensor, Compose, Normalize
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers import StableDiffusionPipeline
import torch.utils.checkpoint
from modules.relax_lora import LoRALinearLayer, LoraLoaderMixin
from modules.utils.lora_utils import unet_lora_state_dict, text_encoder_lora_state_dict
from modules.hypernet import HyperDream

pretrain_model_path = "stable-diffusion-models/realisticVisionV40_v40VAE"
hypernet_model_path = "projects/AIGC/experiments2/hypernet/CelebA-HQ-10k-pretrain2/checkpoint-250000"
reference_image_path = "projects/AIGC/dataset/FFHQ_test/00019.png"
output_dir = "projects/AIGC/experiments2/rank_relax"

# Parameter Settings
train_text_encoder = True
patch_mlp = False
down_dim = 160
up_dim = 80
rank_in = 1  # hypernet output
rank_out = 4  # rank relax output
# vit_model_name = "vit_base_patch16_224"
# vit_model_name = "vit_huge_patch14_clip_224"
vit_model_name = "vit_huge_patch14_clip_336"

t0 = time.time()
# TODO: 1.load predicted lora weights
pipe = StableDiffusionPipeline.from_pretrained(pretrain_model_path, torch_dtype=torch.float32)
# state_dict, network_alphas = pipe.lora_state_dict(lora_model_path)
pipe.to("cuda")
unet = pipe.unet
text_encoder = pipe.text_encoder

# TODO: 2.Create rank_relaxed LoRA
unet_lora_parameters = []
unet_lora_linear_layers = []
print("Create a combined LoRA consisted of Frozen LoRA and Trainable LoRA.")
for i, (attn_processor_name, attn_processor) in enumerate(unet.attn_processors.items()):
    print("unet.attn_processor->%d:%s" % (i, attn_processor_name), attn_processor)
    # attn_processor_name: mid_block.attentions.0.transformer_blocks.0.attn1.processor
    # Parse the attention module.
    attn_module = unet
    for n in attn_processor_name.split(".")[:-1]:
        attn_module = getattr(attn_module, n)
    print("attn_module:", attn_module)

    # Set the `lora_layer` attribute of the attention-related matrices.
    attn_module.to_q.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank_out
        )
    )
    attn_module.to_k.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank_out
        )
    )
    attn_module.to_v.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank_out
        )
    )
    attn_module.to_out[0].set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_out[0].in_features, out_features=attn_module.to_out[0].out_features,
            rank=rank_out,
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
                rank=rank_out,
            )
        )
        attn_module.add_v_proj.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.add_v_proj.in_features,
                out_features=attn_module.add_v_proj.out_features,
                rank=rank_out,
            )
        )
        unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

        unet_lora_linear_layers.append(attn_module.add_k_proj.lora_layer)
        unet_lora_linear_layers.append(attn_module.add_v_proj.lora_layer)

# The text encoder comes from 🤗 transformers, so we cannot directly modify it.
# So, instead, we monkey-patch the forward calls of its attention-blocks.
if train_text_encoder:
    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
    # if patch_mlp is True, the finetuning will cover the text encoder mlp, otherwise only the text encoder attention, total lora is (12+12)*4=96
    # if state_dict is not None, the frozen linear will be initializaed.
    text_lora_parameters, text_encoder_lora_linear_layers = LoraLoaderMixin._modify_text_encoder(text_encoder,
                                                                                                 state_dict=None,
                                                                                                 dtype=torch.float32,
                                                                                                 rank=rank_out,
                                                                                                 patch_mlp=patch_mlp)
    # print(text_encoder_lora_linear_layers)

# total loras
lora_linear_layers = unet_lora_linear_layers + text_encoder_lora_linear_layers if train_text_encoder else unet_lora_linear_layers
print("========================================================================")
t1 = time.time()

# TODO: 3.Convert rank_lora to a standard LoRA
print("Create Hypernet...")
if vit_model_name == "vit_base_patch16_224":
    img_encoder_model_name = "vit_base_patch16_224"
    ref_img_size = 224
    mean = [0.5000]
    std = [0.5000]
elif vit_model_name == "vit_huge_patch14_clip_224":
    img_encoder_model_name = "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k"
    ref_img_size = 224
    mean = [0.4815, 0.4578, 0.4082]
    std = [0.2686, 0.2613, 0.2758]
elif vit_model_name == "vit_huge_patch14_clip_336":
    img_encoder_model_name = "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
    ref_img_size = 336
    mean = [0.4815, 0.4578, 0.4082]
    std = [0.2686, 0.2613, 0.2758]
else:
    raise ValueError("%s does not supports!" % vit_model_name)

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
    weight_dim=(up_dim + down_dim) * rank_in,
)
hypernetwork.set_lilora(lora_linear_layers)

if os.path.isdir(hypernet_model_path):
    path = os.path.join(hypernet_model_path, "hypernetwork.bin")
    weight_dict = torch.load(path)
    sd = weight_dict['hypernetwork']
    hypernetwork.load_state_dict(sd)
else:
    weight_dict = torch.load(hypernet_model_path['hypernetwork'])
    sd = weight_dict['hypernetwork']
    hypernetwork.load_state_dict(sd)

for i, lilora in enumerate(lora_linear_layers):
    seed = weight_dict['aux_seed_%d' % i]
    down_aux = weight_dict['down_aux_%d' % i]
    up_aux = weight_dict['up_aux_%d' % i]

print(f"Hypernet weights loaded from: {hypernet_model_path}")

hypernetwork = hypernetwork.to("cuda")
hypernetwork = hypernetwork.eval()

ref_img = Image.open(reference_image_path).convert("RGB")
ref_img = hypernet_transposes(ref_img).unsqueeze(0).to("cuda")
# warmup
_ = hypernetwork(ref_img)

t2_0 = time.time()
weight, weight_list = hypernetwork(ref_img)
print("weight>>>>>>>>>>>:", weight.shape, weight)
t2_1 = time.time()

t111 = time.time()
# convert down and up weights to linear layer as LoRALinearLayer
for i, (weight, lora_layer) in enumerate(zip(weight_list, lora_linear_layers)):
    seed = weight_dict['aux_seed_%d' % i]
    down_aux = weight_dict['down_aux_%d' % i]
    up_aux = weight_dict['up_aux_%d' % i]
    # reshape weight
    down_weight, up_weight = weight.split([down_dim * rank_in, up_dim * rank_in], dim=-1)
    down_weight = down_weight.reshape(rank_in, -1)
    up_weight = up_weight.reshape(-1, rank_in)
    # make weight, matrix multiplication
    down = down_weight @ down_aux
    up = up_aux @ up_weight
    lora_layer.down.weight.data.copy_(down.to(torch.float32))
    lora_layer.up.weight.data.copy_(up.to(torch.float32))
t222 = time.time()

print("Convert to standard LoRA...")
for lora_linear_layer in lora_linear_layers:
    lora_linear_layer = lora_linear_layer.to("cuda")
    lora_linear_layer.convert_to_standard_lora()
print("========================================================================")
t2 = time.time()

# TODO: 4.Save standard LoRA
print("Save standard LoRA...")
unet_lora_layers_to_save = unet_lora_state_dict(unet)
text_encoder_lora_layers_to_save = None
if train_text_encoder:
    text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(text_encoder, patch_mlp=patch_mlp)

LoraLoaderMixin.save_lora_weights(
    save_directory=output_dir,
    unet_lora_layers=unet_lora_layers_to_save,
    text_encoder_lora_layers=text_encoder_lora_layers_to_save,
)

t3 = time.time()

print("Successfully save LoRA to: %s" % (output_dir))
print("load pipeline: %f" % (t1 - t0))
print("load hypernet: %f" % (t2_0 - t1))
print("hypernet inference: %f" % (t2_1 - t2_0))
print("copy weight: %f" % (t222 - t111))

print("rank relax: %f" % (t2 - t2_1))
print("model save: %f" % (t3 - t2))
print("total time: %f" % (t3 - t0))
print("==================================complete======================================")
