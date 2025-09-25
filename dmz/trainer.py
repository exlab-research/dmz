"""
Trainer from [guided-diffusion](https://github.com/openai/guided-diffusion) and
[DDPM-IP](https://github.com/forever208/DDPM-IP) adjusted to handle additional input `z` and finetuning.
"""

import blobfile as bf
import functools
import os

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import logger, dist_util
from guided_diffusion import train_util, fp16_util
from guided_diffusion.resample import LossAwareSampler


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = fp16_util._flatten_dense_tensors(
            [fp16_util.param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape).type(master_param.data.dtype)

fp16_util.model_grads_to_master_grads = model_grads_to_master_grads


class TrainLoop(train_util.TrainLoop):
    def __init__(self, find_unused_parameters, finetune_attention, finetune_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available() and find_unused_parameters:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=find_unused_parameters or finetune_attention,  # fix for VQ
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model

        if finetune_attention:

            for name, param in self.model.named_parameters():
                if name in finetune_names:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            params = list([param for name, param in self.model.named_parameters()])
            self.opt = AdamW(params, weight_decay=self.weight_decay)
            self.mp_trainer = fp16_util.MixedPrecisionTrainer(
                model=self.model,
                use_fp16=self.use_fp16,
                fp16_scale_growth=1e-3,
            )

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name, param.requires_grad)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            z = self.model.encode(micro)
            z = z.latent

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                z=z,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()

            train_util.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)
