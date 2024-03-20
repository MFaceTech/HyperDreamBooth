import time

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers import StableDiffusionPipeline
import torch.utils.checkpoint
from modules.relax_lora import LoRALinearLayer, LoraLoaderMixin
from modules.utils.lora_utils import unet_lora_state_dict, text_encoder_lora_state_dict

t0 = time.time()

pretrain_model_path="stable-diffusion-models/realisticVisionV40_v40VAE"
lora_model_path = "projects/AIGC/lora_model_test"
output_dir = "projects/AIGC/experiments2/rank_relax"

train_text_encoder = True
patch_mlp = False

# TODO: 1.load predicted lora weights
pipe = StableDiffusionPipeline.from_pretrained(pretrain_model_path, torch_dtype=torch.float32)
state_dict, network_alphas = pipe.lora_state_dict(lora_model_path)
pipe.to("cuda")

unet = pipe.unet
text_encoder = pipe.text_encoder
# print(state_dict.keys())
# unet.up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.lora.down.weight
# unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora.down.weight
# text_encoder.text_model.encoder.layers.11.self_attn.out_proj.lora_linear_layer.down.weight
# text_encoder.text_model.encoder.layers.11.mlp.fc1.lora_linear_layer.down.weight

# TODO: 2.Create rank_relaxed LoRA and initialize the froze linear layer
rank = 4  # the relax lora rank is 4.
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
    print("attn_module:",attn_module)

    # Set the `lora_layer` attribute of the attention-related matrices.
    attn_module.to_q.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank
        )
    )
    attn_module.to_k.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank
        )
    )
    attn_module.to_v.set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank
        )
    )
    attn_module.to_out[0].set_lora_layer(
        LoRALinearLayer(
            in_features=attn_module.to_out[0].in_features,
            out_features=attn_module.to_out[0].out_features,
            rank=rank,
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

    # Set predicted weights to frozen lora
    # static_dict: unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora.up.weight,
    # static_dict: unet.up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight
    # attn_processor_name: mid_block.attentions.0.transformer_blocks.0.attn1.processor
    for layer_name in ['to_q', 'to_k', 'to_v', 'to_out']:
        attn_processor_name = attn_processor_name.replace('.processor', '')
        if layer_name == 'to_out':
            layer = getattr(attn_module, layer_name)[0].lora_layer
            down_key = "unet.%s.%s.0.lora.down.weight" % (attn_processor_name, layer_name)
            up_key = "unet.%s.%s.0.lora.up.weight" % (attn_processor_name, layer_name)
        else:
            layer = getattr(attn_module, layer_name).lora_layer
            down_key = "unet.%s.%s.lora.down.weight" % (attn_processor_name, layer_name)
            up_key = "unet.%s.%s.lora.up.weight" % (attn_processor_name, layer_name)
        # copy weights
        layer.down.weight.data.copy_(state_dict[down_key].to(torch.float32))
        layer.up.weight.data.copy_(state_dict[up_key].to(torch.float32))
    print("unet attention lora initialized!")

    if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
        attn_module.add_k_proj.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.add_k_proj.in_features,
                out_features=attn_module.add_k_proj.out_features,
                rank=rank,
            )
        )
        attn_module.add_v_proj.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.add_v_proj.in_features,
                out_features=attn_module.add_v_proj.out_features,
                rank=rank,
            )
        )
        unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())

        unet_lora_linear_layers.append(attn_module.add_k_proj.lora_layer)
        unet_lora_linear_layers.append(attn_module.add_v_proj.lora_layer)

        for layer_name in ['add_k_proj', 'add_v_proj']:
            attn_processor_name = attn_processor_name.replace('.processor', '')
            layer = getattr(attn_module, layer_name).lora_layer
            down_key = "unet.%s.%s.lora.down.weight" % (attn_processor_name, layer_name)
            up_key = "unet.%s.%s.lora.up.weight" % (attn_processor_name, layer_name)
            # copy weights
            layer.down.weight.data.copy_(state_dict[down_key].to(torch.float32))
            layer.up.weight.data.copy_(state_dict[up_key].to(torch.float32))
            print("unet add_proj lora initialized!")

# The text encoder comes from 🤗 transformers, so we cannot directly modify it.
# So, instead, we monkey-patch the forward calls of its attention-blocks.
if train_text_encoder:
    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
    # if patch_mlp is True, the finetuning will cover the text encoder mlp, otherwise only the text encoder attention, total lora is (12+12)*4=96
    # if state_dict is not None, the frozen linear will be initializaed.
    text_lora_parameters, text_encoder_lora_linear_layers = LoraLoaderMixin._modify_text_encoder(text_encoder, state_dict, dtype=torch.float32, rank=rank, patch_mlp=patch_mlp)
    # print(text_encoder_lora_linear_layers)


# TODO: 3.Convert rank_lora to a standard LoRA
print("Convert rank_lora to a standard LoRA...")
lora_linear_layers = unet_lora_linear_layers + text_encoder_lora_linear_layers \
            if train_text_encoder else unet_lora_linear_layers

for lora_linear_layer in lora_linear_layers:
    lora_linear_layer = lora_linear_layer.to("cuda")
    lora_linear_layer.convert_to_standard_lora()


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

t1 = time.time()

print("Successfully save LoRA to: %s" % (output_dir))
print("time elapsed: %f"%(t1-t0))
print("==================================complete======================================")

