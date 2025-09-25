import argparse
import os
from tqdm import tqdm

import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid

from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict
from guided_diffusion import logger, dist_util

from ddpm_train import (
    model_defaults,
    diffusion_defaults,
    create_model,
    create_gaussian_diffusion,
    log_number_of_params,
)
from dmz.datasets import load_data

from ddpm_eval import save_images
from bridge_train import MappingNetwork


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        model_path="",
        a_data_dir="",
        b_latents_train_path="",
        b_latents_test_path="",
        bridge_path="",
        start_idx=0,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    dir_path = args.model_path.replace("models", "eval")
    logger.configure(dir_path)

    logger.log("creating model and diffusion...")
    args.in_channels = 3 if "edges64" in args.a_data_dir else 1
    model = create_model(**args_to_dict(args, model_defaults().keys()))
    diffusion_args = args_to_dict(args, diffusion_defaults().keys())
    diffusion = create_gaussian_diffusion(**diffusion_args)
    log_number_of_params(model)

    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=True)
    logger.log(f"loaded checkpoint: {args.model_path}")
    logger.log(f"timesteps: {args.timestep_respacing}")
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()
    latents_train = torch.load(args.b_latents_train_path, weights_only=False)["latents"][args.start_idx :]
    latents_test = torch.load(args.b_latents_test_path, weights_only=False)["latents"]

    loaded = torch.load(args.bridge_path, weights_only=False)
    bridge_args = loaded["args"]

    bridge = MappingNetwork(
        latents_test.shape[-1] // 2,
        args.latent_dim,
        hidden_dim=bridge_args["hidden_dim"],
        num_layers=bridge_args["layers_num"],
    ).to(dist_util.dev())
    bridge.load_state_dict(loaded["model"])
    bridge = bridge.to(dist_util.dev())
    bridge.eval()
    logger.log(f"Bridge loaded {args.bridge_path}")

    a_data = load_data(
        data_dir=args.a_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        class_cond=True,
    )

    from ddpm_eval import extend_bernoulli_latent

    num_images = 10
    num_samples = 10
    images = list()
    with torch.no_grad():
        for batch in a_data:
            img_a, cond = batch
            break
        for i in tqdm(range(num_images), total=num_images):
            a = img_a[i : i + 1]
            if a.shape[1] == 1:
                a = a.repeat(1, 3, 1, 1)
            z_b = latents_test[i : i + 1].to(dist_util.dev())
            z_b = (z_b > 0.5).float().reshape(z_b.shape[0], -1, 2)[:, :, 1]
            z_a = (bridge(z_b) > 0.5).float()
            z_a = extend_bernoulli_latent(z_a)
            z_a = torch.cat([z_a.clone() for _ in range(num_samples)], dim=0)
            sample = diffusion.p_sample_loop(
                model,
                (z_a.shape[0], args.in_channels, args.image_size, args.image_size),
                z_a,
                clip_denoised=args.clip_denoised,
            )
            if sample.shape[1] == 1:
                sample = sample.repeat(1, 3, 1, 1)
            images.append(a.detach().cpu())
            images.append(torch.ones_like(a).detach().cpu())
            images.append(sample.detach().cpu())
        images = torch.cat(images, dim=0)
        save_images(images, nrow=num_samples + 2, filename=f"bridge_T{diffusion.num_timesteps}.png")

    if num_samples > 0:
        all_images = list()
        num_batches = (args.num_samples // args.batch_size) + 1
        with torch.no_grad():
            for i in tqdm(range(num_batches), total=num_batches):
                idx = i * args.batch_size
                z_b = latents_train[idx : idx + args.batch_size].to(dist_util.dev())
                z_b = (z_b > 0.5).float().reshape(z_b.shape[0], -1, 2)[:, :, 1]
                z_a = (bridge(z_b) > 0.5).float()
                z_a = extend_bernoulli_latent(z_a)
                sample = diffusion.p_sample_loop(
                    model,
                    (z_a.shape[0], args.in_channels, args.image_size, args.image_size),
                    z_a,
                    clip_denoised=args.clip_denoised,
                )

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()

                all_images.extend([sample.cpu().numpy()])
                logger.log(f"created {len(all_images) * args.batch_size} samples")

        if len(all_images) > 0:
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            shape_str = "x".join([str(x) for x in arr.shape])
            T = diffusion.num_timesteps
            if args.start_idx > 0:
                out_path = os.path.join(logger.get_dir(), f"bridge_idx{args.start_idx}_T{T}_{shape_str}.npz")
            else:
                out_path = os.path.join(logger.get_dir(), f"bridge_T{T}_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)


if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()
