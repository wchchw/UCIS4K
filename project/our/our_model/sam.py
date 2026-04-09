import torch.utils.checkpoint
from torch import nn
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import (
    SamVisionEncoderOutput, SamVisionLayer, SamPatchEmbeddings, SamVisionNeck
)
from typing import Optional, Tuple, Union



class UAViTLayer(SamVisionLayer):
    @torch.no_grad()
    def layer_norm1_no_grad(self, x):
        return self.layer_norm1(x)

    @torch.no_grad()
    def layer_norm2_no_grad(self, x):
        return self.layer_norm2(x)

    @torch.no_grad()
    def attn_no_grad(self, hidden_states, output_attentions=False):
        return self.attn(hidden_states, output_attentions)

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
            adapter: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.FloatTensor]:

        # Store input as residual for later addition
        residual = hidden_states
        # Normalize input states without gradient calculations
        hidden_states = self.layer_norm1_no_grad(hidden_states)
        # If a window size is set, partition the image into smaller windows
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
        # Apply attention mechanism
        hidden_states, attn_weights = self.attn_no_grad(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # Restore the original shape if windows were used
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        hidden_states = residual + hidden_states
        residual2 = hidden_states
        layernorm_output = self.layer_norm2_no_grad(hidden_states)
        residual3 = self.mlp(layernorm_output)
        if getattr(adapter, 'channel_adapter', False):
            residual3 = 0.2*residual3 * adapter.channel_adapter(residual2) + 0.8*residual3
        hidden_states = hidden_states + residual3
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class UAViTEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config  # Store the configuration
        self.image_size = config.image_size  # Store image size from config

        # Initialize patch embeddings
        self.patch_embed = SamPatchEmbeddings(config)

        # Initialize positional embeddings if absolute positions are used
        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        # Initialize layers
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = UAViTLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        # Initialize the "neck" of the model
        self.neck = SamVisionNeck(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    @torch.no_grad()
    def patch_embed_no_grad(self, x):
        return self.patch_embed(x)

    @torch.enable_grad()
    def patch_embed_grad(self, x):
        return self.patch_embed(x)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            adapter: Optional[torch.nn.Module] = None,
            patch_embed_grad: Optional[bool] = False,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate pixel_values input
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Embed pixel values with or without gradients based on flag
        if patch_embed_grad:
            hidden_states = self.patch_embed_grad(pixel_values)
        else:
            hidden_states = self.patch_embed_no_grad(pixel_values)

        # Add positional embeddings if present
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # Initialize containers for outputs if needed
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Process each layer
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:  #
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions,
                    adapter=getattr(adapter, f'adapter_{i}', None)
                )

            hidden_states = layer_outputs[0]

            # Store attention weights if needed
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Store final hidden states if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Process through the neck of the model
        hidden_states = self.neck(hidden_states)

        # Prepare outputs based on return type preference
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
