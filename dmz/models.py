import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.autoencoders.vae import Encoder, EncoderTiny, VectorQuantizer
from diffusers.utils import BaseOutput


class EncoderOutput(BaseOutput):
    latent: torch.Tensor = None
    hidden: torch.Tensor = None
    loss: torch.Tensor = None


class DecoderOutput(BaseOutput):
    sample: torch.Tensor = None


class AutoEncoderOutput(BaseOutput):
    sample: torch.Tensor = None
    latent: torch.Tensor = None
    hidden: torch.Tensor = None
    loss: torch.Tensor = None


def build_encoder(in_channels, channels, out_channels, kernel_size=3):
    modules = []
    for h_dim in channels + [out_channels]:
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels=h_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        in_channels = h_dim
    return nn.Sequential(*modules)


def build_decoder(in_channels, channels, out_channels, kernel_size=3):
    modules = []
    for h_dim in channels[::-1] + [in_channels]:
        modules.append(
            nn.Sequential(
                nn.LeakyReLU(),
                nn.ConvTranspose2d(
                    out_channels,
                    out_channels=h_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(h_dim),
            )
        )
        out_channels = h_dim
    return nn.Sequential(*modules)


class BernoulliEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        in_channels,
        channels,
        tau=1,
        embeddings_2d=False,
    ):
        super(BernoulliEncoder, self).__init__()
        channels, out_channels = channels[:-1], channels[-1]
        self.latent_dim = latent_dim
        self.embeddings_2d = embeddings_2d

        self.layers = build_encoder(in_channels, channels, out_channels)

        size = self.layers(torch.rand(1, in_channels, image_size, image_size)).shape[2]
        self.proj = nn.Linear(out_channels * size * size, 2 * latent_dim)
        self.tau = tau

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.layers(x)
        h = torch.flatten(h, start_dim=1)
        h = self.proj(h)
        h = h.reshape(batch_size, -1, 2)
        hidden = F.gumbel_softmax(h, tau=self.tau, hard=False, dim=-1)
        z = F.gumbel_softmax(h, tau=self.tau, hard=not self.training, dim=-1)
        if not self.embeddings_2d:
            z = z.reshape(batch_size, -1)
            hidden = hidden.reshape(batch_size, -1)
        return EncoderOutput(latent=z, hidden=hidden)


class DiffDecoder(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        in_channels,
        out_channels,
        channels,
        down_block_types,
        up_block_types,
        mid_block_type,
        layers_per_block,
        attention_head_dim,
        resnet_time_scale_shift,
        dropout,
        encoder_type,
        encoder_channels,
        z_via_cross_att=False,
        z_with_t=False,
        z_concat=False,
        use_fp16=False,
        unet_kwargs=None,
    ):
        super(DiffDecoder, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.encoder_type = encoder_type.lower()
        self.z_via_cross_att = z_via_cross_att
        self.z_with_t = z_with_t
        self.temb_dim = channels[0] * 4

        if latent_dim > 0:
            encoder_cls = {
                "bernoulli": BernoulliEncoder,
            }[encoder_type.lower()]
            self.encoder = encoder_cls(
                image_size,
                latent_dim,
                in_channels=in_channels,
                channels=encoder_channels,
            )
            latent_dim = self.encoder(torch.randn(1, in_channels, image_size, image_size)).latent.shape[-1]
        else:
            self.encoder = None

        z_args = {}
        z_args["cross_attention_dim"] = latent_dim if z_via_cross_att else 1

        if z_with_t:
            z_args["class_embeddings_concat"] = z_concat
            z_args["class_embed_type"] = "simple_projection"
            z_args["projection_class_embeddings_input_dim"] = latent_dim

        if unet_kwargs:
            adjust_mid_block = mid_block_type == "UNetMidBlock2D"
        else:
            adjust_mid_block = False
            unet_kwargs = {}

        self.cross_attention_dim = z_args["cross_attention_dim"]
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            mid_block_type=mid_block_type,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            resnet_time_scale_shift=resnet_time_scale_shift,
            **z_args,
            **unet_kwargs,
        )
        if adjust_mid_block:
            # To match huggingface DDPM checkpoints
            self.unet.mid_block = UNetMidBlock2D(
                in_channels=channels[-1],
                temb_channels=channels[0] * 4,
                dropout=dropout,
                resnet_eps=unet_kwargs["norm_eps"],
                resnet_act_fn="silu",
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim,
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                continue
            module.half()

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        for module in self.modules():
            module.float()

    def encode(self, x):
        if self.encoder:
            h = x.type(self.dtype)
            out = self.encoder(h)
            return EncoderOutput(
                latent=out.latent.type(x.dtype),
                hidden=out.hidden.type(x.dtype),
                loss=out.loss,
            )
        else:
            return EncoderOutput(latent=None, hidden=None, loss=None)

    def forward(self, x, t, z=None):
        h = x.type(self.dtype)
        z = z.type(self.dtype) if z is not None else None
        hidden_states = (
            z
            if self.z_via_cross_att
            else torch.ones(x.shape[0], self.cross_attention_dim, device=x.device, dtype=self.dtype)
        )
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(1)
        if self.z_with_t:
            z = z.reshape(z.shape[0], -1)
        else:
            z = None
        sample = self.unet(h, timestep=t, encoder_hidden_states=hidden_states, class_labels=z)["sample"].type(x.dtype)
        return sample
