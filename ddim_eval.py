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

from ddpm_eval import save_images, extract_latent_codes, create_sampler, create_argparser

def _extract_into_tensor(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def sample_and_save(
    model,
    diffusion,
    z_sampler,
    in_channels,
    image_size,
    num_samples,
    batch_size,
    clip_denoised,
    prefix,
    eta=0,
):
    all_images = []
    n_iters = (num_samples // batch_size) + 1 if num_samples > 0 else 0
    with torch.no_grad():
        for _ in tqdm(range(n_iters), total=n_iters):
            if len(all_images) * batch_size >= num_samples:
                break
            z = z_sampler.sample(batch_size, dist_util.dev()) if z_sampler is not None else None
            sample = diffusion.ddim_sample_loop(
                model,
                (batch_size, in_channels, image_size, image_size),
                z,
                clip_denoised=clip_denoised,
                eta=eta,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            all_images.extend([sample.cpu().numpy()])
            logger.log(f"created {len(all_images) * batch_size} samples")

    if len(all_images) > 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[:num_samples]
        shape_str = "x".join([str(x) for x in arr.shape])
        T = diffusion.num_timesteps
        out_path = os.path.join(logger.get_dir(), f"{prefix}_eta{eta}_T{T}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        arr = torch.from_numpy(arr[:64]).permute(0, 3, 1, 2).float() / 255.0
        arr = 2 * arr - 1
        save_images(arr, nrow=8, filename=f"{prefix}_eta{eta}_T{diffusion.num_timesteps}.png")



def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    dir_path = args.model_path.replace("models", "eval")
    logger.configure(dir_path)

    logger.log("creating model and diffusion...")
    args.in_channels = 1 if "edges64" in args.train_data_dir else 3
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

    if model.encoder is not None:
        train_latents, train_classes = extract_latent_codes(
            args.train_data_dir, args.batch_size, args.image_size, model
        )
        out_path = os.path.join(logger.get_dir(), f"latents_train.pt")
        torch.save({"latents": train_latents, "classes": train_classes}, out_path)
        logger.log(f"Latents saved to {out_path}")

        test_latents, test_classes = extract_latent_codes(args.test_data_dir, args.batch_size, args.image_size, model)
        out_path = os.path.join(logger.get_dir(), f"latents_test.pt")
        torch.save({"latents": test_latents, "classes": test_classes}, out_path)
        logger.log(f"Latents saved to {out_path}")

        if args.sampler_name in ["dataset", "random"]:
            if args.encoder_type == "bernoulli":
                train_latents = (train_latents > 0.5).float()
            elif args.encoder_type == "vq" and args.sampler_name == "dataset":
                train_latents, _ = extract_latent_codes(
                    args.train_data_dir, args.batch_size, args.image_size, model, hidden=False
                )
            else:
                pass

        z_sampler = create_sampler(
            args.sampler_name, args.sampler_path, args.encoder_type, args.model_path, train_latents
        )

    else:
        z_sampler = None

    model.eval()
    print(diffusion.betas)
    if args.num_samples > 0:
        logger.log(f"Sampling {args.num_samples} images")
        prefix = f"ddim_samples"
        if model.encoder is not None:
            prefix += f"_{args.sampler_name}"
        sample_and_save(
            model,
            diffusion,
            z_sampler,
            args.in_channels,
            args.image_size,
            args.num_samples,
            args.batch_size,
            args.clip_denoised,
            prefix)

if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()
