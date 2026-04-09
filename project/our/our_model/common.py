import torch
import math
from torch import nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.dist import is_main_process
from peft import get_peft_config, get_peft_model
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from .sam import UAViTEncoder

class DeepDifferentialConv(nn.Module):
    def __init__(self, channels):
        super(DeepDifferentialConv, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.pointwise = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, x1, x2):
        diff1 = x1 - x2
        diff = self.conv(diff1)
        attn1 = self.pointwise(diff)
        attn = self.sigmoid(attn1)
        out = x1 * attn
        return out


class ChannelBalanceAdapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, change=False) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, embedding_dim*2, 1, bias=False)
        self.fc2 = nn.Conv2d(embedding_dim*2, embedding_dim, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.change_channel = change

    def forward(self, x):
        if self.change_channel:
            x = x.permute(0, 3, 1, 2).contiguous()
            avg_out = self.avg_pool(x)
            ref_out, _ = torch.max(x, dim=1, keepdim=True)
            ref_out0 = ref_out.repeat(1, x.shape[1], 1, 1)
            ref_avg_out = self.avg_pool(ref_out0)
            ref_out1 = self.Sigmoid(self.fc2(self.act(self.fc1(ref_avg_out - avg_out))))
            return ref_out1.permute(0, 2, 3, 1)

class UAViTBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 use_channel_adapter=True,
                 ):
        super().__init__()

        if use_channel_adapter:
            self.channel_adapter = ChannelBalanceAdapter(embed_dim, change=True)


@MODELS.register_module()
class ChannelViTAdapters(BaseModule):

    def __init__(self,
                 adapter_layer,
                 embed_dim,
                 use_channel_adapter=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.adapter_layer = adapter_layer
        for idx in adapter_layer:
            self.add_module(
                f'adapter_{idx}',
                UAViTBlock(embed_dim, use_channel_adapter)
            )


class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.GELU) -> None:
        super().__init__()
        self.act = act_layer()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv3 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(output_dim, output_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(output_dim, output_dim, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv3(x) + self.conv5(x) + self.conv7(x)
        return self.act(self.bn2(x))


@MODELS.register_module(force=True)
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register_module()
class UCISSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class UCISSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


#
@MODELS.register_module()
class UCISSamVisionEncoder(BaseModule):
    def __init__(
            self,
            hf_pretrain_name,  # The name or path to the Hugging Face pretrained model
            extra_config,  # Additional configuration settings to update the SAM configuration
            peft_config=None,  # Configuration for parameter-efficient fine-tuning (PEFT)
            init_cfg=None,  # Configuration for initializing the module, usually includes path to the checkpoint
    ):
        # Initialize the base module class from which this class inherits
        BaseModule.__init__(self, init_cfg=init_cfg)

        # Load the configuration for the SAM (Structured Attention Model) from a pre-trained model
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config

        # Update the SAM configuration with any extra configuration provided
        if extra_config is not None:
            sam_config.update(extra_config)

        # Initialize the vision encoder using the UAViTEncoder class with the updated configuration
        vision_encoder = UAViTEncoder(sam_config)

        # If an initial configuration is provided and includes a checkpoint, load the checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(
                vision_encoder,
                init_cfg.get('checkpoint'),  # Get the checkpoint file path from the init_cfg
                map_location='cpu',  # Load the checkpoint into CPU memory
                revise_keys=[(r'^module\.', ''),
                             (r'^vision_encoder\.', '')])  # Regex to adjust the key names in the state dict

        # If a PEFT configuration is provided, configure the vision encoder for parameter-efficient fine-tuning
        if peft_config is not None and isinstance(peft_config, dict):
            # Default PEFT settings
            config = {
                "peft_type": "LORA",  # Type of PEFT, LORA (Low-Rank Adaptation)
                "r": 16,  # Rank of adaptation
                'target_modules': ["qkv"],  # Target modules in the transformer to apply LORA
                "lora_alpha": 32,  # Scaling factor for the low-rank matrices
                "lora_dropout": 0.05,  # Dropout rate for the LORA layers
                "bias": "none",  # Bias settings for the LORA adaptation
                "inference_mode": False,  # Whether to run in inference mode
            }
            config.update(peft_config)  # Update default config with the provided PEFT settings
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder,
                                                 peft_config)  # Apply the PEFT settings to the vision encoder

            # Print trainable parameters if this is the main process
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder

        # Mark the vision encoder as initialized
        self.vision_encoder.is_init = True

    def init_weights(self):
        if is_main_process():
            print('the vision encoder has been initialized')

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class UCISSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(
            self,
            hf_pretrain_name,
            extra_config=None,
            init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)
