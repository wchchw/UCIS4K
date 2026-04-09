import copy
import einops
import numpy as np
from typing import List, Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmcv.cnn import build_norm_layer, ConvModule
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.task_modules import SamplingResult
from mmdet.structures import SampleList, DetDataSample
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding
from transformers.models.sam.modeling_sam import SamVisionEncoderOutput
from .common import MultiScaleConv, ChannelBalanceAdapter, DeepDifferentialConv
import pywt



@MODELS.register_module()
class UCISAnchor(MaskRCNN):
    def __init__(
            self,
            shared_image_embedding,
            adapter=None,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.adapter = False
        if adapter is not None:
            self.adapter = MODELS.build(adapter)
        self.decoder_freeze = decoder_freeze
        self.frozen_modules = []
        self.DeepDifferentialConv = DeepDifferentialConv(256)
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1280, 640, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
        )
        self.con1x1 = nn.Sequential(
            nn.Conv2d(640, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

        )
        self.con1x11 = torch.nn.Conv2d(in_channels=256 + 640, out_channels=256, kernel_size=1)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def fourier_transform_K(self, image):

        F = torch.fft.fft2(image, dim=(-2, -1))

        # Compute the amplitude (magnitude) spectrum of the Fourier coefficients
        amplitude = torch.abs(F)
        K = 1000
        # Flatten the amplitude for global top-K selection
        amplitude_flat = amplitude.view(-1)
        topk_values, topk_indices = torch.topk(amplitude_flat, K)

        # Create a mask to mark these top-K frequency components
        mask = torch.zeros_like(amplitude_flat, dtype=torch.bool)
        mask[topk_indices] = True
        F_masked = F.clone()
        F_masked.view(-1)[mask] = 0

        img_a = torch.fft.ifft2(F_masked, dim=(-2, -1))
        img_a2 = img_a.real

        return img_a2

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        transformed_images = []
        # Apply Fourier-based reconstruction
        for img in batch_inputs:
            img_reconstructed = self.fourier_transform_K(img)
            transformed_images.append(img_reconstructed)
        batch_inputs_transformed = torch.stack(transformed_images)
        if self.adapter:
            vision_outputs = self.backbone(batch_inputs, adapter=self.adapter)
        else:
            vision_outputs = self.backbone(batch_inputs)
        transformed_outputs = self.backbone(batch_inputs_transformed)

        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            transformed_embeddings = transformed_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[-1]
            transformed_embeddings = transformed_outputs[-1]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        x, all_coeffs_H = self.neck((vision_hidden_states))
        #high frequency
        all_coeffs_H_abs = torch.abs(all_coeffs_H[..., 0]) + torch.abs(all_coeffs_H[..., 1]) + torch.abs(all_coeffs_H[..., 2])
        all_coeffs_H_sqrt = torch.sqrt(all_coeffs_H[..., 0]**2 + all_coeffs_H[..., 1]**2 + all_coeffs_H[..., 2]**2)
        all_coeffs_H1 = all_coeffs_H_sqrt + all_coeffs_H_abs
        high_freq_energies = torch.split(all_coeffs_H1, 1280, dim=1)
        total_high_freq_energy = sum(high_freq_energies)
        ratios = [high_freq_energy / total_high_freq_energy for high_freq_energy in high_freq_energies]
        all_coeffs_H3 = torch.zeros_like(high_freq_energies[0])
        for ratio, tensor in zip(ratios, high_freq_energies):
            all_coeffs_H3 += ratio * tensor
        all_coeffs_H4 = self.up(all_coeffs_H3)
        all_coeffs_H5 = self.con1x1(all_coeffs_H4)

        combined_features = image_embeddings + all_coeffs_H5
        decamouflaged_features = self.DeepDifferentialConv(combined_features, transformed_embeddings)

        return x, decamouflaged_features, image_positional_embeddings


    def _dummy_rois(self, batch_inputs: Tensor, num_rois: int = 100) -> Tensor:
        """Build fixed dummy RoIs for FLOPs/params analysis.

        get_flops.py traces `forward(mode='tensor')` and needs a valid forward path.
        The exact RoI coordinates do NOT affect parameter counting, and only provide a
        stable compute graph for FLOPs tracing.
        """
        B, _, H, W = batch_inputs.shape
        device = batch_inputs.device
        rois = torch.zeros((B * num_rois, 5), device=device, dtype=torch.float32)
        rois[:, 0] = torch.arange(B, device=device).repeat_interleave(num_rois)
        rois[:, 1] = 0.0
        rois[:, 2] = 0.0
        rois[:, 3] = float(W - 1)
        rois[:, 4] = float(H - 1)
        return rois

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList = None):
        """Tensor mode forward for mmengine analysis tools (e.g., tools/analysis_tools/get_flops.py).

        Your training/inference uses `loss()`/`predict()` (overridden below) and is
        unaffected. This path is only for tracing FLOPs/params.
        """
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        # RPN forward (NO NMS/postprocess; tracing-friendly)
        rpn_cls_scores, rpn_bbox_preds = self.rpn_head(x)
        outs = []
        outs.extend(list(rpn_cls_scores))
        outs.extend(list(rpn_bbox_preds))

        # RoI head forward with dummy RoIs (tracing-friendly)
        rois = self._dummy_rois(batch_inputs, num_rois=100)

        if getattr(self.roi_head, 'with_bbox', False):
            bbox_results = self.roi_head._bbox_forward(x, rois)
            outs.append(bbox_results['cls_score'])
            outs.append(bbox_results['bbox_pred'])

        if getattr(self.roi_head, 'with_mask', False):
            # NOTE: HF SAM mask decoder uses torch._shape_as_tensor in some versions, which
            # creates CPU shape tensors during tracing and can trigger CUDA/CPU device mismatch
            # in repeat_interleave. For FLOPs/params analysis (torch.jit tracing), we skip the
            # mask branch to keep get_flops.py working. Params are still counted correctly.
            if not torch.jit.is_tracing():
                mask_results = self.roi_head._mask_forward(
                    x, rois,
                    image_embeddings=image_embeddings,
                    image_positional_embeddings=image_positional_embeddings,
                )
                outs.append(mask_results['mask_preds'])

        return tuple(outs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        device = batch_inputs.device
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
        losses = dict()
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)  # rpn loss#loss_rpn_cls和loss_rpn_bbox加入losses中

        roi_losses = self.roi_head.loss(
            x, rpn_results_list, batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


@MODELS.register_module()
class UCISFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        inputs_original = inputs

        if hasattr(self, 'feature_aggregator'):
            #low frequency, high frequency
            x_LL, all_coeffs_H = self.feature_aggregator(inputs_original)
            x = x_LL
        else:
            x = inputs
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x, all_coeffs_H



@MODELS.register_module()
class UCISSimpleFPNHead(BaseModule):
    def __init__(self,
                 backbone_channel: int,  # Number of channels in the input feature map from the backbone
                 in_channels: List[int],  # List of channels for each input feature map
                 out_channels: int,  # Number of channels for each output feature map
                 num_outs: int,  # Number of output feature maps to generate
                 conv_cfg: OptConfigType = None,  # Configuration for convolution layers
                 norm_cfg: OptConfigType = None,  # Configuration for normalization layers
                 act_cfg: OptConfigType = None,  # Configuration for activation layers
                 init_cfg: MultiConfig = None) -> None:  # Initialization configuration for module weights
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        # Define several FPN layers with different specifications
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            # Transpose convolution to upsample
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],  # Normalization layer
            nn.GELU(),  # Activation function GELU
            nn.ConvTranspose2d(self.backbone_channel // 2, self.backbone_channel // 4, 2,
                               2))  # Further transpose convolution to upsample
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())  # Identity layer to pass features unchanged
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling to downsample the feature map

        # Lists to hold lateral and output convolutional modules
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # Initialize lateral and fpn convolutions for each input feature map
        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        """Forward function to compute the output feature maps.

        Args:
            input (Tensor): Input feature map from the backbone.

        Returns:
            tuple: A tuple of output feature maps, each as a 4D tensor.
        """
        # Apply each FPN block to the input to generate intermediate feature maps
        inputs = [self.fpn1(input), self.fpn2(input), self.fpn3(input), self.fpn4(input)]
        #
        # Apply lateral convolutions to the intermediate feature maps
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Generate final output feature maps using FPN convolutions
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # Add extra levels to the outputs by applying max pooling, if necessary
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@MODELS.register_module()
class UCISFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,
            select_layers,
            hidden_channels=64,
            out_channels=256,
            mean_alpha=0.8,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.select_layers = select_layers
        self.alpha = mean_alpha

        self.ca_adapter = ChannelBalanceAdapter(in_channels[0], 0.0625, act_layer=nn.ReLU)

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                MultiScaleConv(self.in_channels[i_layer], hidden_channels)
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(1280 * 4, out_channels, 1),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def wavedec2_transform(self, tensor, wavelet='db1', level=1):
        """
        DWT
        """
        tensor_np = tensor.cpu().detach().numpy()
        coeffs = pywt.wavedec2(tensor_np, wavelet, level=level)
        return coeffs

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]
        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(inputs[i_layer])


        device = next(self.parameters()).device
        batch_size = len(features)
        all_coeffs_L = []
        all_coeffs_H = []

        for i in range(batch_size):
            feature = features[i].contiguous().float()
            coeffs_L = []
            coeffs_H = []

            for j in range(feature.size(0)):
                single_coeffs = self.wavedec2_transform(feature[j], level=1)
                coeffs_L.append(single_coeffs[0])

                LH, HL, HH = single_coeffs[1]
                high_freq_combined = np.stack([LH, HL, HH], axis=-1)
                coeffs_H.append(high_freq_combined)

            all_coeffs_L.append(torch.tensor(np.stack(coeffs_L)).float())
            all_coeffs_H.append(torch.tensor(np.stack(coeffs_H)).float())


        all_coeffs_L = torch.cat(all_coeffs_L, dim=1).view(feature.size(0), -1, 32, 32).to(device)

        all_coeffs_H = torch.cat(all_coeffs_H, dim=1).view(feature.size(0), -1, 32, 32, 3).to(device)
        L_up = self.upsample(self.fusion_conv1(all_coeffs_L))
        L_relu = self.relu(self.fusion_conv2(L_up) + L_up)
        return L_relu, all_coeffs_H




@MODELS.register_module()
class UCISPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
            self,
            with_extra_pe=False,  # Flag to indicate whether to use extra positional encoding
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,  # Normalization flag for positional encoding
            )

            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def _mask_forward(self,
                      x: Tuple[Tensor],  # Input feature maps
                      rois: Tensor = None,  # Regions of Interest
                      pos_inds: Optional[Tensor] = None,  # Positive sample indices
                      bbox_feats: Optional[Tensor] = None,  # Features from bounding boxes
                      image_embeddings=None,  # Image embeddings
                      image_positional_embeddings=None  # Image positional embeddings
                      ) -> dict:
        assert ((rois is not None) ^  # Ensure either ROIs or bbox features are provided
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            # Extract features for the given ROIs
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                # Process through the shared head if specified
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        # Predict masks and IOU predictions using the mask head
        mask_preds, iou_predictions = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None,
        )
        # Compile results into a dictionary
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],  # Input feature maps
                  sampling_results: List[SamplingResult],  # Sampling results
                  bbox_feats: Tensor,  # Bounding box features
                  batch_gt_instances: InstanceList,  # Ground truth instances
                  image_embeddings=None,
                  image_positional_embeddings=None
                  ) -> dict:
        if not self.share_roi_extractor:
            # Extract positive ROIs if ROI extractor is not shared
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                print('no pos rois')
                # Return zero loss if no positive ROIs
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            # Forward mask processing
            mask_results = self._mask_forward(
                x, pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                # Collect positive and negative indicators
                pos_inds.append(torch.ones(res.pos_priors.shape[0], device=device, dtype=torch.uint8))
                pos_inds.append(torch.zeros(res.neg_priors.shape[0], device=device, dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        # Calculate mask loss and targets
        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        # Update results with calculated loss
        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def loss(self, x: Tuple[Tensor],  # Input feature maps
             rpn_results_list: InstanceList,  # Results from RPN
             batch_data_samples: List[DetDataSample],  # Data samples
             image_embeddings=None,
             image_positional_embeddings=None
             ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)  # Ensure the lengths match
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, 'extra_pe'):
            # Apply extra positional encoding if present
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # Assign ground truth and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # Calculate losses for bounding box head if present
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # Forward and calculate losses for mask head if present
        if self.with_mask:
            mask_results = self.mask_loss(
                x, sampling_results, bbox_results['bbox_feats'], batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update(mask_results['loss_mask'])

        return losses

    def predict_mask(
            self,
            x: Tuple[Tensor],
            batch_img_metas: List[dict],
            results_list: InstanceList,
            rescale: bool = False,
            image_embeddings=None,
            image_positional_embeddings=None,
    ) -> InstanceList:

        # Don't need to consider augmentation during test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            # Handle cases with no detections
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        # Forward mask processing
        mask_results = self._mask_forward(
            x, mask_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings)

        mask_preds = mask_results['mask_preds']
        # Split batch mask predictions back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # Handle rescaling and prediction finalization
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                image_embeddings=None,
                image_positional_embeddings=None,
                ) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if hasattr(self, 'extra_pe'):
            # Apply extra positional encoding if present
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return results_list


@MODELS.register_module()
class UCISPrompterAnchorMaskHead(FCNMaskHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: ConfigType = dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None,
            *args,
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        # Build mask decoder from provided config
        self.mask_decoder = MODELS.build(mask_decoder)

        # Set up prompt encoder based on mask decoder configuration
        prompt_encoder = dict(
            type='UCISSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()

        # Initialize no mask embed from prompt encoder
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1

        # Define point embedding based on the input channels, feature size, and sine-cosine usage
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * roi_feat_size ** 2 // 4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self,
                x,
                image_embeddings,
                image_positional_embeddings,
                roi_img_ids=None,
                ):
        # Extract batch sizes and dimensions
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c', n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())

        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image,
                                       torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device,
                                                   dtype=num_roi_per_image.dtype)])

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0],
                                                                                 image_embedding_size[1])
        # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        # assert False, f"{low_res_masks.shape}"
        return low_res_masks, iou_predictions

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear',
                                align_corners=False).squeeze(1)

        scale_factor_w, scale_factor_h = img_meta['scale_factor']
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]

        h, w = img_meta['ori_shape']
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask
