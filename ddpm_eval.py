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
from dmz.datasets import load_data, CELEBA_index_to_label
from dmz.sampler import DatasetSampler, EmpiricalSampler, RandomSampler, PixelSNAILSampler, ConstSampler, NormalSampler
from dmz.sampler import extend_bernoulli_latent


def interpolate_z_continuous(z, steps):
    z0, z1 = z[0], z[1]
    z = torch.zeros(steps + 2, z.shape[-1]).to(z.device)
    z[0] = z0
    z[-1] = z1
    for i in range(1, steps+1):
        step_size = i / (steps+1)
        z[i] = z0 * (1- step_size) + z1 * step_size
    return z


def interpolate_z(z, steps, order=0):
    z = z.reshape(z.shape[0], -1, 2)[:, :, 1]
    z0, z1 = z[0], z[1]
    if order > 0:
        perm = torch.randperm(z.shape[-1])
        inv_perm = torch.argsort(perm)
        z0, z1 = z0[perm], z1[perm]
    else:
        inv_perm = None
    z = torch.zeros(steps + 2, z.shape[-1]).to(z.device)
    diff_idx = torch.arange(len(z0)).to(z.device)[z0 != z1]
    step_size = len(diff_idx) / (steps + 1)
    for i in range(steps):
        idx = diff_idx[: int((i + 1) * step_size)]
        z[i + 1] = z0.clone()
        z[i + 1, idx] = z1[idx]
    z[0] = z0
    z[-1] = z1
    if order > 0:
        z = z[:, inv_perm]
    z = extend_bernoulli_latent(z)
    return z


def interpolate_z_smooth(x1, x2, model, num_steps):
    """
    Interpolate between binary latent codes z1 and z2 with smooth transitions,
    guided by a symmetric blend of q(z|x1) and q(z|x2).

    Args:
        z1, z2 : np.ndarray of shape (d,)
            Binary latent codes from x1 and x2
        x1, x2 : inputs corresponding to z1 and z2
        encode : function (x) -> np.ndarray of shape (d,)
            Returns Bernoulli probs for each bit (e.g., model.encode)
        measure : str
            Distance measure to use (default: 'cross_entropy')

    Returns:
        path : list of np.ndarray
            Sequence of binary latent vectors from z1 to z2
    """
    z1 = model.encode(x1.unsqueeze(0)).latent.squeeze(0)
    z2 = model.encode(x2.unsqueeze(0)).latent.squeeze(0)
    z1 = z1.reshape(-1, 2)[:,1]
    z2 = z2.reshape(-1, 2)[:,1]
    path = [z1.clone()]
    current = z1.clone()
    total_flips = torch.sum(z1 != z2)
    steps_done = 0

    while not (current == z2).all():
        diffs = torch.where(current != z2)[0]

        best_flip = None
        min_cost = float('inf')

        alpha = steps_done / total_flips
        probs1 = model.encode(x1.to(dist_util.dev()).unsqueeze(0)).hidden.squeeze(0)
        probs2 = model.encode(x2.to(dist_util.dev()).unsqueeze(0)).hidden.squeeze(0)
        probs1 = probs1.reshape(-1, 2)[:,1]
        probs2 = probs2.reshape(-1, 2)[:,1]
        assert probs1.min() >= 0 and probs1.max() <= 1, probs1
        assert probs2.min() >= 0 and probs2.max() <= 1, probs2
        probs = (1 - alpha) * probs1 + alpha * probs2  # blended q(z|x)
        assert probs.min() >= 0 and probs.max() <= 1, (alpha, probs, current == z2)
        for i in diffs:
            candidate = current.clone()
            candidate[i] = 1 - candidate[i]  # flip bit i
            # Binary cross entropy
            eps = 1e-8
            p = probs
            z = candidate
            cost = -torch.sum(z * torch.log(p + eps) + (1 - z) * torch.log(1 - p + eps))
            if cost < min_cost:
                min_cost = cost
                best_flip = i

        current[best_flip] = 1 - current[best_flip]
        path.append(current.clone())
        steps_done += 1

    assert (path[-1] == z2).all()
    n = len(path) - 2
    step = n // num_steps
    if n % num_steps != 0:
        step += 1
    path = path[:1] + path[1:-1:step] + path[-1:]
    # assert len(path) == num_steps + 2, (len(path), num_steps, n, n//num_steps, step)
    path = torch.stack(path).to(x1.device)
    path = extend_bernoulli_latent(path)
    return path


def save_images(images, nrow, filename):
    grid = make_grid(0.5 * (images + 1), nrow=nrow, padding=2, pad_value=1)
    grid_image = torchvision.transforms.ToPILImage()(grid)
    out_path = os.path.join(logger.get_dir(), filename)
    grid_image.save(out_path)


def extract_latent_codes(data_dir, batch_size, image_size, model, hidden=True):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=True,
        deterministic=True,
        random_flip=False,
        random_crop=False,
    )
    num_images = len(os.listdir(data_dir))
    latents, classes = list(), list()
    num_batches = num_images // batch_size
    if num_images % batch_size != 0:
        num_batches += 1
    with torch.no_grad():
        for batch in tqdm(data, total=num_batches):
            img, cond = batch
            z = model.encode(img.to(dist_util.dev()))
            latent = z.hidden if hidden else z.latent
            latents.append(latent.detach().cpu())
            if "y" in cond:
                classes.append(cond["y"].detach().cpu())
            if len(latents) == num_batches:
                break

    latents = torch.cat(latents, dim=0)[:num_images]
    classes = torch.cat(classes, dim=0)[:num_images] if len(classes) > 0 else None
    return latents, classes


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
):
    all_images = []
    n_iters = (num_samples // batch_size) + 1 if num_samples > 0 else 0
    with torch.no_grad():
        for _ in tqdm(range(n_iters), total=n_iters):
            if len(all_images) * batch_size >= num_samples:
                break
            z = z_sampler.sample(batch_size, dist_util.dev()) if z_sampler is not None else None
            sample = diffusion.p_sample_loop(
                model,
                (batch_size, in_channels, image_size, image_size),
                z,
                clip_denoised=clip_denoised,
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
        out_path = os.path.join(logger.get_dir(), f"{prefix}_T{T}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        arr = torch.from_numpy(arr[:64]).permute(0, 3, 1, 2).float() / 255.0
        arr = 2 * arr - 1
        save_images(arr, nrow=8, filename=f"{prefix}_T{diffusion.num_timesteps}.png")


def sample_and_save_intermediate(
    data_dir,
    model,
    diffusion,
    in_channels,
    image_size,
    batch_size,
    clip_denoised,
    prefix,
):
    batch_size = 500
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        class_cond=True,
    )
    num_images = len(os.listdir(data_dir))
    num_batches = num_images // batch_size
    if num_images % batch_size != 0:
        num_batches += 1
    all_images = []
    all_zs = []
    with torch.no_grad():
        for batch in tqdm(data, total=num_batches):
            img, cond = batch
            img = img.to(dist_util.dev())
            z = model.encode(img).latent
            sample = diffusion.p_sample_loop(
                model,
                (z.shape[0], in_channels, image_size, image_size),
                z,
                clip_denoised=clip_denoised,
                return_intermediate=True,
            )
            all_zs.extend([z.cpu().numpy()])
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 1, 3, 4, 2)
            sample = sample.contiguous()

            all_images.extend([sample.cpu().numpy()])
            logger.log(f"created {len(all_images) * batch_size} samples")
            if len(all_images) * batch_size >= num_images:
                break

    if len(all_images) > 0:
        arr = np.concatenate(all_images, axis=1)
        arr = arr[:num_images]
        shape_str = "x".join([str(x) for x in arr.shape])
        T = diffusion.num_timesteps
        out_path = os.path.join(logger.get_dir(), f"intermediate_{prefix}_T{T}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        zs = np.concatenate(all_zs, axis=0)
        zs = zs[:num_images]
        shape_str = "x".join([str(x) for x in zs.shape])
        out_path = os.path.join(logger.get_dir(), f"intermediate_z_{prefix}_T{T}_{shape_str}.npz")
        np.savez(out_path, zs)
        
        ts = list(range(10)) + list(range(0, 101))[10::10]
        for i in range(arr.shape[0]):
            arr_i = arr[i]
            arr_i = torch.from_numpy(arr_i[:64]).permute(0, 3, 1, 2).float() / 255.0
            arr_i = 2 * arr_i - 1
            save_images(arr_i, nrow=8, filename=f"intermediate_{prefix}_T{diffusion.num_timesteps}_{i}_{ts[i]}.png")


def calculate_elbo(data, model, diffusion, batch_size, clip_denoised, num_images):
    bpd = list()
    num_batches = num_images // batch_size
    if num_images % batch_size != 0:
        num_batches += 1
    with torch.no_grad():
        for batch in tqdm(data, total=num_batches):
            x_start, cond = batch
            x_start = x_start.to(dist_util.dev())
            z = model.encode(x_start).hidden if model.encoder_type != "vq" else model.encode(x_start).latent
            out = diffusion.calc_bpd_loop(
                model,
                x_start,
                z=z,
                clip_denoised=clip_denoised,
            )
            bpd.append(out["total_bpd"])
            if len(bpd) >= num_batches:
                break
    bpd = torch.cat(bpd)[:num_images].mean().item()
    return bpd


def nll_eval(
    model,
    diffusion_args,
    diffusion_steps,
    num_samples,
    batch_size,
    image_size,
    train_data_dir,
    test_data_dir,
    clip_denoised,
):
    batch_size = min(batch_size, num_samples)
    diffusion_args["timestep_respacing"] = [diffusion_steps]
    curr_diffusion = create_gaussian_diffusion(**diffusion_args)
    data = load_data(
        data_dir=train_data_dir,
        batch_size=batch_size,
        image_size=image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )
    train_bpd = calculate_elbo(
        data,
        model,
        curr_diffusion,
        batch_size,
        clip_denoised,
        num_images=num_samples,
    )
    train_bpd_str = f"BPD train T'={curr_diffusion.num_timesteps} N={num_samples}: {train_bpd:.4f}"
    logger.log(train_bpd_str)

    data = load_data(
        data_dir=test_data_dir,
        batch_size=batch_size,
        image_size=image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
    )
    test_bpd = calculate_elbo(
        data,
        model,
        curr_diffusion,
        batch_size,
        clip_denoised,
        num_images=num_samples,
    )
    test_bpd_str = f"BPD test T'={curr_diffusion.num_timesteps} N={num_samples}: {test_bpd:.4f}"
    logger.log(test_bpd_str)

    out_path = os.path.join(logger.get_dir(), f"nll.txt")
    with open(out_path, "a") as f:
        f.write(f"{train_bpd_str}\n{test_bpd_str}\n")
    logger.log(f"NLL eval saved to {out_path}")


def create_sampler(sampler_name, sampler_path, encoder_type, model_path, latents):

    if sampler_name == "dataset":
        return DatasetSampler(latents)
    elif sampler_name.startswith("dataset"):
        n = int(sampler_name[len("dataset") :])
        return DatasetSampler((latents[:n] > 0.5).float())
    elif sampler_name == "random":
        return RandomSampler(encoder_type, latents.shape[1:], latents.max().long().item() + 1)
    elif sampler_name == "empirical":
        return EmpiricalSampler(latents)
    elif sampler_name == "normal":
        return NormalSampler(latents)
    elif sampler_name == "pixelsnail":
        logger.log(f"loading pixelsnail {sampler_path}")
        return PixelSNAILSampler(encoder_type, sampler_path, dist_util.dev())
    elif sampler_name == "const":
        if encoder_type == "bernoulli":
            val = latents[:1].detach().cpu()
            val = (val > 0.5).float()
            return ConstSampler(val, latents.shape[1:])
        else:
            assert False
    else:
        assert False, sampler_name


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        nll_num_samples=0,
        nll_timesteps=1000,
        sampler_name="dataset",
        batch_size=16,
        model_path="",
        train_data_dir="",
        test_data_dir="",
        save_intermediate=False,
        sampler_path="",
        classifier_path="",
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

    model.eval()

    if args.nll_num_samples > 0:
        nll_eval(
            model,
            diffusion_args,
            args.nll_timesteps,
            args.nll_num_samples,
            args.batch_size,
            args.image_size,
            args.train_data_dir,
            args.test_data_dir,
            args.clip_denoised,
        )

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

    if args.num_samples > 0:
        logger.log(f"Sampling {args.num_samples} images")
        prefix = f"samples_{args.sampler_name}" if model.encoder is not None else "samples"
        sample_and_save(
            model,
            diffusion,
            z_sampler,
            args.in_channels,
            args.image_size,
            args.num_samples,
            args.batch_size,
            args.clip_denoised,
            prefix,
        )

    if model.encoder is None:
        return

    if args.save_intermediate:
        logger.log(f"Intermediate images")
        assert diffusion.num_timesteps == 100, diffusion.num_timesteps
        sample_and_save_intermediate(
            args.test_data_dir,
            model,
            diffusion,
            args.in_channels,
            args.image_size,
            args.batch_size,
            args.clip_denoised,
            prefix="test",
        )
        sample_and_save_intermediate(
            args.train_data_dir,
            model,
            diffusion,
            args.in_channels,
            args.image_size,
            args.batch_size,
            args.clip_denoised,
            prefix="train",
        )
        logger.log("Intermediate images saved")

    data = load_data(
        data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        class_cond=True,
    )

    logger.log("Reconstruction from z")
    num_samples = 5 if args.batch_size > 64 else 5
    num_images = 10
    img = None
    samples = None
    mse = list()
    with torch.no_grad():
        for batch in data:
            img, cond = batch
            break
        img = img[:num_images].to(dist_util.dev())
        z = model.encode(img).hidden if model.encoder_type != "vq" else model.encode(img).latent
        z = (z > 0.5).float()
        if z is not None:
            z = torch.stack([z[i].clone() for i in range(z.shape[0]) for _ in range(num_samples)])
            z = z.to(img.device).float()

        samples = diffusion.p_sample_loop(
            model,
            (z.shape[0], args.in_channels, args.image_size, args.image_size),
            z=z,
            clip_denoised=args.clip_denoised,
        )
        target = (
            torch.stack([img[i].clone() for i in range(img.shape[0]) for _ in range(num_samples)])
            .to(dist_util.dev())
            .float()
        )
        mse.append(torch.mean((samples - target) ** 2).item())
    grid = []
    for i in range(img.shape[0]):
        grid.append(img[i: i + 1].cpu())
        grid.append(torch.ones_like(img[0:1]).cpu())
        grid.append(samples[i * num_samples: (i + 1) * num_samples].cpu())
    grid = torch.cat(grid, dim=0)
    save_images(grid, nrow=num_samples + 2, filename=f"reconstruct_from_z_T{diffusion.num_timesteps}.png")
    logger.log("Done")


    logger.log("Examples of denoising")
    timestamps = [0.1 * float(t) for t in range(1, 10, 2)]
    T = int(args.timestep_respacing)
    num_images = 16
    timestamps = [int(t * T) for t in timestamps]
    with torch.no_grad():
        for batch in data:
            for t in timestamps:
                img, cond = batch
                img = img.to(dist_util.dev())[:num_images]
                t_tensor = torch.tensor([t] * img.shape[0], device=dist_util.dev())
                img_noisy = diffusion.q_sample(img, t_tensor)
                z = model.encode(img).hidden if model.encoder_type != "vq" else model.encode(img).latent
                out = diffusion.ddim_sample(model, img_noisy, t_tensor, z, clip_denoised=args.clip_denoised)
                pred_img = out["pred_xstart"]

                images = torch.cat([img, img_noisy, pred_img], dim=0)
                save_images(
                    images,
                    nrow=8,
                    filename=f"denoise_t{t:04}_random_T{args.timestep_respacing}.png",
                )
            break
    logger.log("Done")

    logger.log("Interpolations with varying noise levels")
    with torch.no_grad():
        steps = 5
        num_images = 1
        for batch in data:
            img, cond = batch
            break
        for j in range(num_images):
            img_i = img.to(dist_util.dev())  # [:num_images]
            img_i = img_i[j : j + 2]
            z = model.encode(img_i).hidden if model.encoder_type != "vq" else model.encode(img_i).latent
            z = (z > 0.5).float()
            z = interpolate_z(z, steps=steps).to(img_i.device)
            images = list()
            for t in timestamps:
                t_tensor = torch.tensor([t] * img_i.shape[0], device=dist_util.dev())
                img_noisy = diffusion.q_sample(img_i, t_tensor)
                alphas = torch.linspace(0, 1, steps + 2)
                img_noisy = torch.stack([(1 - alpha) * img_noisy[0] + alpha * img_noisy[1] for alpha in alphas])

                pred_img = diffusion.p_sample_loop(
                    model, img_noisy.shape, noise=img_noisy, z=z, clip_denoised=args.clip_denoised, just_t=t
                )

                images.append(img_i[:2])
                images.append(img_noisy[:1])
                images.append(torch.ones_like(img_i[:1]))
                images.append(pred_img)
            images = torch.cat(images, dim=0)
            t_str = "_".join([f"{t}" for t in timestamps])
            save_images(
                images,
                nrow=4 + steps + 2,
                filename=f"denoise_t{t_str}_inter_T{args.timestep_respacing}_{j}.png",
            )
    logger.log("Done")

    logger.log("Impact of T")
    Ts = [1000, 500, 200, 100, 50, 20, 10, 5]
    num_images = 10
    for k in range(1):
        images = []
        z = z_sampler.sample(num_images, device=dist_util.dev()) if z_sampler is not None else None
        shape = (num_images, args.in_channels, args.image_size, args.image_size)
        noise = torch.randn(*shape, device=dist_util.dev())
        for T in tqdm(Ts, total=len(Ts)):
            diffusion_args["timestep_respacing"] = [T]
            curr_diffusion = create_gaussian_diffusion(**diffusion_args)
            with torch.no_grad():
                sample = curr_diffusion.p_sample_loop(
                    model,
                    shape,
                    z=z.clone() if z is not None else None,
                    noise=noise.clone(),
                    clip_denoised=args.clip_denoised,
                )
                images.append(sample)

        sufix = "_".join([str(t) for t in Ts])
        images = torch.cat(images)
        save_images(images, nrow=num_images, filename=f"T{sufix}_{k}.png")


    logger.log("Done")
    mse = np.mean(mse)
    logger.log(f"MSE {mse:.5f}")
    out_path = os.path.join(logger.get_dir(), "mse.txt")
    with open(out_path, "w") as f:
        f.write(f"MSE T={diffusion.num_timesteps}: {mse:.5f}\n")
    logger.log(f"MSE saved to {out_path}")

    num_images = 10
    num_steps = 5
    orders = 2
    images = []
    logger.log("Discrete interpolations of z")
    with torch.no_grad():
        for batch in data:
            img, cond = batch
            img = img[:2].to(dist_util.dev())

            for order in range(orders):
                z = model.encode(img).hidden if model.encoder_type != "vq" else model.encode(img).latent
                z = (z > 0.5).float()
                z = interpolate_z(z, steps=num_steps, order=order).to(img.device)
                if z is not None:
                    noise = torch.randn_like(img[0]).to(z.device)
                    noise = torch.stack([noise.clone() for _ in z])
                else:
                    noise = None

                samples = diffusion.p_sample_loop(
                    model,
                    (z.shape[0], args.in_channels, args.image_size, args.image_size),
                    z=z,
                    noise=noise,
                    clip_denoised=args.clip_denoised,
                )
                images.append(img)
                images.append(torch.ones_like(img[:1]))
                images.append(samples)
            if len(images) > num_images * 3 * orders:
                break

        grid = torch.cat(images, dim=0)
        save_images(grid, nrow=num_steps + 4 + 1, filename=f"interpolations_T{diffusion.num_timesteps}.png")
    logger.log("Done")

    num_images = 5
    num_steps = 6
    logger.log("Continuous interpolations in z")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data), total=num_images):
            img, cond = batch
            img = img[:2].to(dist_util.dev())
            z = model.encode(img).latent
            z = interpolate_z_continuous(z, steps)
            noise = torch.randn_like(img[0]).to(z.device)
            noise = torch.stack([noise.clone() for _ in z])
            samples = diffusion.p_sample_loop(
                model,
                (z.shape[0], args.in_channels, args.image_size, args.image_size),
                z=z,
                noise=noise,
                clip_denoised=args.clip_denoised,
            )
            images = list()
            images.append(img)
            images.append(torch.ones_like(img[:1]))
            images.append(samples)
            grid = torch.cat(images, dim=0)
            save_images(
                grid, nrow=num_steps + 3, filename=f"interpolations_continuous_{i:02}_T{diffusion.num_timesteps}.png"
            )
            if i + 1 == num_images:
                break
    logger.log("Done")

    data = load_data(
        data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        class_cond=True,
    )
    num_images = 5
    num_steps = 8
    logger.log("Smooth interpolations in z")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data), total=num_images):
            img, cond = batch
            img = img[:2].to(dist_util.dev())
            cond = {k: v[:2].to(dist_util.dev()) for k, v in cond.items()}

            z = interpolate_z_smooth(img[0], img[1], model, num_steps)
            noise = torch.randn_like(img[0]).to(z.device)
            noise = torch.stack([noise.clone() for _ in z])
            samples = diffusion.p_sample_loop(
                model,
                (z.shape[0], args.in_channels, args.image_size, args.image_size),
                z=z,
                noise=noise,
                clip_denoised=args.clip_denoised,
            )

            images = list()
            images.append(img)
            images.append(torch.ones_like(img[:1]))
            images.append(samples)
            grid = torch.cat(images, dim=0)
            save_images(
                grid, nrow=num_steps + 5, filename=f"interpolations_smooth_{i:02}_T{diffusion.num_timesteps}.png"
            )
            if i + 1 == num_images:
                break

    logger.log("Done")

    if args.classifier_path and os.path.exists(args.classifier_path):
        logger.log("Features manipulation with classifiers")
        with torch.no_grad():
            loaded = torch.load(args.classifier_path, map_location="cpu")
            num_images = 5
            for batch in data:
                img, cond = batch
                img = img[2: num_images + 2].to(dist_util.dev())
                y = cond["y"][2: num_images + 2]
                break

            z = model.encode(img).latent
            for t in [0.9, 1.0]:
                t = int(diffusion.num_timesteps * t)
                t_tensor = torch.tensor([t] * img.shape[0], device=dist_util.dev())
                img_noisy = diffusion.q_sample(img, t_tensor) if t < diffusion.num_timesteps else torch.rand_like(img).to(img.device)

                for label in tqdm(range(y.shape[-1]), total=y.shape[-1]):
                    classifier_weights = loaded[label]
                    W = classifier_weights["layers.0.weight"].T.to(dist_util.dev())
                    b = classifier_weights["layers.0.bias"].to(dist_util.dev())
                    w_diff = W[:, 1] - W[:, 0]
                    b_diff = b[1] - b[0]
                    w_norm = torch.sqrt((w_diff ** 2).sum())

                    z_copy = z.clone()
                    d = (z_copy * w_diff.unsqueeze(0)).sum(-1) + b_diff
                    d = d / w_norm
                    step = (d.unsqueeze(-1) * w_diff.unsqueeze(0)) / w_norm

                    images = [img, img_noisy]

                    for alpha in [-4, -2, -1, 0, 1, 2, 4]:
                        z_prim = z_copy + alpha * step
                        z_prim = (z_prim > 0.5).float()

                        with torch.no_grad():
                            samples = diffusion.p_sample_loop(
                                model,
                                img_noisy.shape,
                                z=z_prim,
                                noise=img_noisy,
                                clip_denoised=args.clip_denoised,
                                just_t=t,
                            )
                        images.append(samples)
                    grid = torch.cat(images, dim=0)
                    n_col = len(grid) // num_images
                    grid = grid.reshape(n_col, -1, *grid.shape[1:]).permute(1, 0, 2, 3, 4).reshape(-1, *grid.shape[1:])
                    save_images(
                        grid,
                        nrow=n_col,
                        filename=f"labels_T{diffusion.num_timesteps}_t{t}_{label}_{CELEBA_index_to_label[label]}.png",
                    )
        logger.log("Done")


if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()
