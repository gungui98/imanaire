# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
from functools import partial
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from imaginaire.generators.fs_vid2vid import LabelEmbedder
from imaginaire.layers import Conv2dBlock, LinearBlock, Res2dBlock
from imaginaire.model_utils.fs_vid2vid import (extract_valid_pose_labels,
                                               resample)
from imaginaire.utils.data import (get_paired_input_image_channel_number,
                                   get_paired_input_label_channel_number)
from imaginaire.utils.init_weight import weights_init
from imaginaire.utils.speckle import create_mapping


class BaseNetwork(nn.Module):
    r"""vid2vid generator."""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_num_filters(self, num_downsamples):
        r"""Get the number of filters at current layer.

        Args:
            num_downsamples (int) : How many downsamples at current layer.
        Returns:
            output (int) : Number of filters.
        """
        return min(self.max_num_filters,
                   self.num_filters * (2 ** num_downsamples))


class Generator(BaseNetwork):
    r"""vid2vid generator constructor.

    Args:
        gen_cfg (obj): Generator definition part of the yaml config file.
        data_cfg (obj): Data definition part of the yaml config file.
    """

    def __init__(self, gen_cfg, data_cfg):
        super().__init__()
        self.gen_cfg = gen_cfg
        self.data_cfg = data_cfg
        self.num_frames_G = data_cfg.num_frames_G
        # Number of residual blocks in generator.
        self.num_layers = num_layers = getattr(gen_cfg, 'num_layers', 7)
        # Number of downsamplings for previous frame.
        self.num_downsamples_img = getattr(gen_cfg, 'num_downsamples_img', 4)
        # Number of filters in the first layer.
        self.num_filters = num_filters = getattr(gen_cfg, 'num_filters', 32)
        self.max_num_filters = getattr(gen_cfg, 'max_num_filters', 1024)
        self.kernel_size = kernel_size = getattr(gen_cfg, 'kernel_size', 3)
        padding = kernel_size // 2

        # Input data params.
        num_input_channels = get_paired_input_label_channel_number(data_cfg)
        num_img_channels = get_paired_input_image_channel_number(data_cfg)
        aug_cfg = data_cfg.val.augmentations
        if hasattr(aug_cfg, 'center_crop_h_w'):
            crop_h_w = aug_cfg.center_crop_h_w
        elif hasattr(aug_cfg, 'resize_h_w'):
            crop_h_w = aug_cfg.resize_h_w
        else:
            raise ValueError('Need to specify output size.')
        crop_h, crop_w = crop_h_w.split(',')
        crop_h, crop_w = int(crop_h), int(crop_w)
        # Spatial size at the bottle neck of generator.
        self.sh = crop_h // (2 ** num_layers)
        self.sw = crop_w // (2 ** num_layers)

        # Noise vector dimension.
        self.z_dim = getattr(gen_cfg, 'style_dims', 256)
        self.use_segmap_as_input = \
            getattr(gen_cfg, 'use_segmap_as_input', False)

        # Label / image embedding network.
        self.emb_cfg = emb_cfg = getattr(gen_cfg, 'embed', None)
        self.use_embed = getattr(emb_cfg, 'use_embed', 'True')
        self.num_downsamples_embed = getattr(emb_cfg, 'num_downsamples', 5)
        if self.use_embed:
            self.label_embedding = LabelEmbedder(emb_cfg, num_input_channels)

        # At beginning of training, only train an image generator.
        self.temporal_initialized = False
        # Whether to output hallucinated frame (when training temporal network)
        # for additional loss.
        self.generate_raw_output = False

        # Image generation network.
        weight_norm_type = getattr(gen_cfg, 'weight_norm_type', 'spectral')
        activation_norm_type = gen_cfg.activation_norm_type
        activation_norm_params = gen_cfg.activation_norm_params
        if self.use_embed and \
                not hasattr(activation_norm_params, 'num_filters'):
            activation_norm_params.num_filters = 0
        nonlinearity = 'leakyrelu'

        self.base_res_block = base_res_block = partial(
            Res2dBlock, kernel_size=kernel_size, padding=padding,
            weight_norm_type=weight_norm_type,
            activation_norm_type=activation_norm_type,
            activation_norm_params=activation_norm_params,
            nonlinearity=nonlinearity, order='NACNAC')

        # Upsampling residual blocks.
        for i in range(num_layers, -1, -1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            activation_norm_params.partial = self.get_partial(
                i) if hasattr(self, 'get_partial') else False
            layer = base_res_block(self.get_num_filters(i + 1),
                                   self.get_num_filters(i))
            setattr(self, 'up_%d' % i, layer)

        # Final conv layer.
        self.conv_img = Conv2dBlock(num_filters, num_img_channels,
                                    kernel_size, padding=padding,
                                    nonlinearity=nonlinearity, order='AC')

        num_filters = min(self.max_num_filters,
                          num_filters * (2 ** (self.num_layers + 1)))
        self.fc = Conv2dBlock(num_input_channels, num_filters,
                              kernel_size=3, padding=1)

        # Misc.
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = partial(F.interpolate, scale_factor=2)
        self.combine = Conv2dBlock(6, 3, kernel_size=1, stride=1)
        self.noisy_backgrounds = []
        self.init_temporal_network()
        self._init_noisy_background()

    def _init_noisy_background(self, ):
        r"""
        Initialize the noisy background.
        """
        x_map, y_map = create_mapping(600, 400)
        noisy_images = []
        for i in range(100):
            image = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
            image = cv2.resize(image, (600, 400))
            # blur the image
            image = cv2.blur(image, (15, 1))
            # resize
            image = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
            image = image[:, 55:-55]
            image = cv2.resize(image, (512, 512))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # real_image = data["image"]
            # from imaginaire.utils.visualization import tensor2im
            #
            # real_image = tensor2im(real_image)[0]
            # real_image = cv2.resize(real_image, (512, 512))
            # add_weighted_image = cv2.addWeighted(image, 0.5, real_image, 0.5, 0)
            # cv2.imshow("add_weighted_image", add_weighted_image)
            # cv2.waitKey(0)

            # convert to tensor
            image = torchvision.transforms.ToTensor()(image)
            image = image.unsqueeze(0)
            noisy_images.append(image)
        self.noisy_backgrounds = noisy_images

    def get_noisy_background(self, batch_size):
        r"""
        Get the noisy background.
        """

        noisy_backgrounds = []
        for i in range(batch_size):
            noisy_backgrounds.append(random.choice(self.noisy_backgrounds))
        return torch.cat(noisy_backgrounds, dim=0)

    def forward(self, data):
        r"""vid2vid generator forward.

        Args:
           data (dict) : Dictionary of input data.
        Returns:
           output (dict) : Dictionary of output data.
        """
        label = data['label']
        label_prev, img_prev = data['prev_labels'], data['prev_images']
        is_first_frame = img_prev is None
        bs, _, h, w = label.size()

        # Get SPADE conditional maps by embedding current label input.
        cond_maps_now = self.get_cond_maps(label, self.label_embedding)

        # Input to the generator will either be noise/segmentation map (for
        # first frame) or encoded previous frame (for subsequent frames).
        if is_first_frame:
            # First frame in the sequence, start from scratch.
            x_img = F.interpolate(label, size=(self.sh, self.sw))
            x_img = self.fc(x_img)

            # Upsampling layers.
            for i in range(self.num_layers, self.num_downsamples_img, -1):
                j = min(self.num_downsamples_embed, i)
                x_img = getattr(self, 'up_' + str(i))(x_img, *cond_maps_now[j])
                x_img = self.upsample(x_img)
        else:
            # Not the first frame, will encode the previous frame and feed to
            # the generator.
            x_img = self.down_first(img_prev[:, -1])

            # Get label embedding for the previous frame.
            cond_maps_prev = self.get_cond_maps(label_prev[:, -1],
                                                self.label_embedding)

            # Downsampling layers.
            for i in range(self.num_downsamples_img + 1):
                j = min(self.num_downsamples_embed, i)
                x_img = getattr(self, 'down_' + str(i))(x_img,
                                                        *cond_maps_prev[j])
                if i != self.num_downsamples_img:
                    x_img = self.downsample(x_img)

            # Resnet blocks.
            j = min(self.num_downsamples_embed, self.num_downsamples_img + 1)
            for i in range(self.num_res_blocks):
                cond_maps = cond_maps_prev[j] if i < self.num_res_blocks // 2 \
                    else cond_maps_now[j]
                x_img = getattr(self, 'res_' + str(i))(x_img, *cond_maps)

        mask = None

        num_frames_G = self.num_frames_G

        # Main image generation branch.
        for i in range(self.num_downsamples_img, -1, -1):
            # Get SPADE conditional inputs.
            j = min(i, self.num_downsamples_embed)
            cond_maps = cond_maps_now[j]


            # For final output.

            x_img = self.one_up_conv_layer(x_img, cond_maps, i)

        # Final conv layer.
        img_final = torch.tanh(self.conv_img(x_img))

        # add speckle noise to final image

        noisy_background = self.get_noisy_background(img_final.shape[0])
        noisy_background = noisy_background.to(img_final.device)
        combine = torch.cat([img_final, noisy_background], dim=1)
        weight_map = torch.sigmoid(self.combine(combine))
        # force the weight map to be 0.2 to 0.8u
        weight_map = (weight_map - 0.2) / (0.8 - 0.2)
        # img_final = noisy_background * weight_map + img_final * (1 - weight_map)

        # get tensor where labels are 0 or 1
        label_background = label[:, :1] + label[:, 1:2]
        # duplicate to 3 channels
        # label_background = label_background.repeat(1, 3, 1, 1)
        # blur the label background
        # label_background = torchvision.transforms.GaussianBlur(kernel_size=3)(label_background)

        output = dict()
        output['fake_images'] = img_final
        output['fake_occlusion_masks'] = mask
        output['fake_weight_maps'] = weight_map
        output['noise_background'] = noisy_background
        output['label_background'] = label_background
        return output

    def one_up_conv_layer(self, x, encoded_label, i):
        r"""One residual block layer in the main branch.

        Args:
           x (4D tensor) : Current feature map.
           encoded_label (list of tensors) : Encoded input label maps.
           i (int) : Layer index.
        Returns:
           x (4D tensor) : Output feature map.
        """
        layer = getattr(self, 'up_' + str(i))
        x = layer(x, *encoded_label)
        if i != 0:
            x = self.upsample(x)
        return x

    def init_temporal_network(self, cfg_init=None):
        r"""When starting training multiple frames, initialize the
        downsampling network and flow network.

        Args:
            cfg_init (dict) : Weight initialization config.
        """
        # Number of image downsamplings for the previous frame.
        num_downsamples_img = self.num_downsamples_img
        # Number of residual blocks for the previous frame.
        self.num_res_blocks = int(
            np.ceil((self.num_layers - num_downsamples_img) / 2.0) * 2)

        # First conv layer.
        num_img_channels = get_paired_input_image_channel_number(self.data_cfg)
        self.down_first = \
            Conv2dBlock(num_img_channels,
                        self.num_filters, self.kernel_size,
                        padding=self.kernel_size // 2)
        if cfg_init is not None:
            self.down_first.apply(weights_init(cfg_init.type, cfg_init.gain))

        # Downsampling residual blocks.
        activation_norm_params = self.gen_cfg.activation_norm_params
        for i in range(num_downsamples_img + 1):
            activation_norm_params.cond_dims = self.get_cond_dims(i)
            layer = self.base_res_block(self.get_num_filters(i),
                                        self.get_num_filters(i + 1))
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'down_%d' % i, layer)

        # Additional residual blocks.
        res_ch = self.get_num_filters(num_downsamples_img + 1)
        activation_norm_params.cond_dims = \
            self.get_cond_dims(num_downsamples_img + 1)
        for i in range(self.num_res_blocks):
            layer = self.base_res_block(res_ch, res_ch)
            if cfg_init is not None:
                layer.apply(weights_init(cfg_init.type, cfg_init.gain))
            setattr(self, 'res_%d' % i, layer)

    def get_cond_dims(self, num_downs=0):
        r"""Get the dimensions of conditional inputs.

        Args:
           num_downs (int) : How many downsamples at current layer.
        Returns:
           ch (list) : List of dimensions.
        """
        if not self.use_embed:
            ch = [self.num_input_channels]
        else:
            num_filters = getattr(self.emb_cfg, 'num_filters', 32)
            num_downs = min(num_downs, self.num_downsamples_embed)
            ch = [min(self.max_num_filters, num_filters * (2 ** num_downs))]
        return ch

    def get_cond_maps(self, label, embedder):
        r"""Get the conditional inputs.

        Args:
           label (4D tensor) : Input label tensor.
           embedder (obj) : Embedding network.
        Returns:
           cond_maps (list) : List of conditional inputs.
        """
        if not self.use_embed:
            return [label] * (self.num_layers + 1)
        embedded_label = embedder(label)
        cond_maps = [embedded_label]
        cond_maps = [[m[i] for m in cond_maps] for i in
                     range(len(cond_maps[0]))]
        return cond_maps