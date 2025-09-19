import argparse
import datetime
import os

from guided_diffusion import logger, dist_util
from guided_diffusion import train_util
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import space_timesteps
from guided_diffusion.script_util import add_dict_to_argparser, args_to_dict

from dmz.datasets import load_data
from dmz.gaussian_diffusion import SpacedDiffusion
from dmz.models import DiffDecoder
from dmz.trainer import TrainLoop

SAVE_PATH = f"saved/models"

HUGGING_FACE_LEGACY_MODE_PARAMS = {
    "downsample_padding": 0,
    "flip_sin_to_cos": False,
    "freq_shift": 1,
    "norm_eps": 1e-06,
}


def create_model(
    image_size,
    num_channels,
    layers_per_block,
    attention_head_dim,
    in_channels=3,
    dropout=0,
    resnet_time_scale_shift="scale_shift",
    latent_dim=16,
    encoder_type="none",
    cross_attention=False,
    one_cross_attention=-1,
    learn_sigma=False,
    z_concat=False,
    z_via_cross_att=False,
    z_with_t=False,
    use_fp16=False,
    hugging_face_checkpoints_params=False,
):
    unet_legacy_kwargs = None
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
        is_attention = (False, False, False, False, True, False)
        if one_cross_attention >= 0:
            is_cross_attention = [False] * len(is_attention)
            assert is_attention[one_cross_attention]
            is_cross_attention[one_cross_attention] = True
            is_cross_attention = tuple(is_cross_attention)
        else:
            is_cross_attention = is_attention if cross_attention else ([False] * len(is_attention))
        encoder_channels = [32, 64, 128, 256, 512, 1024, 2048]
        # UNet params to match huggingface checkpoints
        if hugging_face_checkpoints_params:
            unet_legacy_kwargs = HUGGING_FACE_LEGACY_MODE_PARAMS

    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
        is_attention = (False, True, True, True)

        if one_cross_attention >= 0:
            is_cross_attention = [False] * len(is_attention)
            assert is_attention[one_cross_attention]
            is_cross_attention[one_cross_attention] = True
            is_cross_attention = tuple(is_cross_attention)
        else:
            is_cross_attention = is_attention if cross_attention else ([False] * len(is_attention))

        encoder_channels = [32, 64, 128, 256, 512]

    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
        is_attention = (False, True, True, False)
        if one_cross_attention >= 0:
            is_cross_attention = [False] * len(is_attention)
            assert is_attention[one_cross_attention]
            is_cross_attention[one_cross_attention] = True
            is_cross_attention = tuple(is_cross_attention)
        else:
            is_cross_attention = is_attention if cross_attention else ([False] * len(is_attention))
        encoder_channels = [32, 64, 128, 256]
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    channels = [num_channels * m for m in channel_mult]
    up_block_types = [None] * len(channel_mult)
    down_block_types = [None] * len(channel_mult)
    for i, (ca, a) in enumerate(zip(is_cross_attention, is_attention)):
        if not a:
            down_block_types[i] = "DownBlock2D"
            up_block_types[len(up_block_types) - 1 - i] = "UpBlock2D"
        elif a and (not ca):
            down_block_types[i] = "AttnDownBlock2D"
            up_block_types[len(up_block_types) - 1 - i] = "AttnUpBlock2D"
        elif a and ca:
            down_block_types[i] = "CrossAttnDownBlock2D"
            up_block_types[len(up_block_types) - 1 - i] = "CrossAttnUpBlock2D"
        else:
            assert False

    mid_block_type = "UNetMidBlock2DCrossAttn" if (cross_attention or (one_cross_attention >= 0)) else "UNetMidBlock2D"

    logger.log(down_block_types, up_block_types, mid_block_type)

    return DiffDecoder(
        image_size=image_size,
        latent_dim=latent_dim,
        in_channels=in_channels,
        out_channels=2 * in_channels if learn_sigma else in_channels,
        channels=channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        mid_block_type=mid_block_type,
        layers_per_block=layers_per_block,
        attention_head_dim=attention_head_dim,
        resnet_time_scale_shift=resnet_time_scale_shift,
        dropout=dropout,
        encoder_type=encoder_type,
        encoder_channels=encoder_channels,
        z_via_cross_att=z_via_cross_att,
        z_with_t=z_with_t,
        z_concat=z_concat,
        unet_kwargs=unet_legacy_kwargs,
        use_fp16=use_fp16,
    )


def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=32,
        in_channels=3,
        layers_per_block=3,
        attention_head_dim=32,
        dropout=0.0,
        resnet_time_scale_shift="scale_shift",
        latent_dim=0,
        encoder_type="none",
        learn_sigma=False,
        z_via_cross_att=False,
        cross_attention=False,
        one_cross_attention=-1,
        z_with_t=False,
        z_concat=False,
        use_fp16=False,
        hugging_face_checkpoints_params=False,
    )
    return res


def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    variance_type="learned_range",
    noise_schedule="cosine",
    timestep_respacing=1000,
    rescale_timesteps=False,
    rescale_learned_sigmas=True,
    input_pertub=0.0,
):
    steps = diffusion_steps
    betas = get_named_beta_schedule(noise_schedule, steps)
    if rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type={
            "learned_range": ModelVarType.LEARNED_RANGE,
            "fixed_small": ModelVarType.FIXED_SMALL,
            "fixed_large": ModelVarType.FIXED_LARGE,
        }[variance_type],
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        input_pertub=input_pertub,
    )


def diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        diffusion_steps=1000,
        variance_type="learned_range",
        noise_schedule="cosine",
        timestep_respacing="",
        rescale_timesteps=False,
        rescale_learned_sigmas=True,
        input_pertub=0.0,
    )
    return res


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def log_number_of_params(model):
    logger.log(f"total number of parameters: {count_params(model)}")
    logger.log(f"number of parameters in unet: {count_params(model.unet)}")
    if model.encoder is not None:
        logger.log(model.encoder)
        logger.log(f"number of parameters in encoder: {count_params(model.encoder)}")
    else:
        logger.log("no encoder")


def main():
    args = create_argparser().parse_args()
    print(args)

    dist_util.setup_dist()
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    if args.save_prefix == "":
        dataset = args.data_dir.split("/")[-1].split("_")[0]
        if args.encoder_type != "none":
            cond_code = f"{int(args.z_via_cross_att)}{int(args.cross_attention)}{int(args.one_cross_attention)}{int(args.z_with_t)}{int(args.z_concat)}-"
        else:
            cond_code = ""
        if args.finetune_path != "":
            cond_code += f"finetune-"
        elif f"finetune-" in args.resume_checkpoint:
            cond_code += f"finetune-"
        dir_path = os.path.join(
            SAVE_PATH,
            f"{dataset}-{args.encoder_type}-{args.latent_dim}-{cond_code}{time}",
        )
    else:
        dir_path = os.path.join(SAVE_PATH, f"{args.save_prefix}-{time}")
    logger.configure(dir_path)
    logger.log("creating data loader...")
    args.in_channels = 1 if "edges64" in args.data_dir else 3
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    model = create_model(**args_to_dict(args, model_defaults().keys()))
    log_number_of_params(model)

    if args.finetune_path:
        assert args.resume_checkpoint == ""
        loaded = dist_util.load_state_dict(args.finetune_path, map_location="cpu")

        # Fix keys in huggingface checkpoints
        def add_prefix(state_dict, prefix="unet."):
            return {k if k.startswith(prefix) else prefix + k: v for k, v in state_dict.items()}

        def replace_keys(state_dict):
            state_dict = {
                k.replace("attentions.0.key.", "attentions.0.to_k.") if "mid_block" in k else k: v
                for k, v in state_dict.items()
            }
            state_dict = {
                k.replace("attentions.0.value.", "attentions.0.to_v.") if "mid_block" in k else k: v
                for k, v in state_dict.items()
            }
            return {
                k.replace(".query.", ".to_q.")
                .replace(".key.", ".to_k.")
                .replace(".value.", ".to_v.")
                .replace(".proj_attn.", ".to_out.0."): v
                for k, v in state_dict.items()
            }

        loaded = add_prefix(loaded)
        loaded = replace_keys(loaded)
        names_curr = [name for name, _ in model.named_parameters()]
        names_loaded = [name for name in loaded.keys()]
        names_curr_only = [names for names in names_curr if names not in names_loaded]
        names_loaded_only = [names for names in names_loaded if names not in names_curr]
        logger.log("Keys only in current model", names_curr_only)
        logger.log("Keys only in loaded model", names_loaded_only)

        for name, params in model.named_parameters():
            if name in names_loaded:
                if params.shape != loaded[name].shape:
                    del loaded[name].shape
                    logger.log(f"Removed {name} from load due to shape mismatch")

        names_not_loaded = names_curr_only
        model.load_state_dict(loaded, strict=False)
        logger.log(f"loaded checkpoint for finetuning {args.finetune_path}")
    else:
        names_not_loaded = None

    model.to(dist_util.dev())
    logger.log(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log(f"creating data loader from {args.data_dir}...")

    logger.log("training...")
    TrainLoop(
        find_unused_parameters=False,
        finetune_attention=args.finetune_attention,
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune_names=names_not_loaded,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        input_pertub=0.0,
        save_prefix="",
        finetune_path="",
        finetune_attention=False,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
