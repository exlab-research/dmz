# On Designing Diffusion Autoencoders for Efficient Generation and Representation Learning

## Installation
Installation is the same as in [guided-diffusion](https://github.com/openai/guided-diffusion) and [DDPM-IP](https://github.com/forever208/DDPM-IP).

We use `python=3.10`, `pytorch=2.6.0+cu118`, `diffusers=0.32.0`  and the guided diffusion module implemented in [DDPM-IP](https://github.com/forever208/DDPM-IP).

## Datasets

Use scripts in `datasets/` to preprocess datasets if needed. Use `unpack_npz.py` to unpack images in npz format into a directory, e.g.
```bash
python unpack_npz.py --npz_path data/CelebA/celeba64_train.npz --output_dir data/CelebA/celeba64_train
```

## Training

### CIFAR-10
```bash
mpiexec -n 1 python ddpm_train.py --data_dir data/cifar10/cifar10_train --input_pertub 0.15 \
    --image_size 32 --use_fp16 True --num_channels 128 --attention_head_dim 32 --layers_per_block 3 \
    --variance_type learned_range --learn_sigma True --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine  \
    --rescale_learned_sigmas True  --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128 \
    --encoder_type bernoulli --latent_dim 16 --z_via_cross_att True  --cross_attention True
```
### CelebA-64
```bash
mpiexec -n 2 python ddpm_train.py --data_dir data/celeba/celeba64_train --input_pertub 0.1  \
    --image_size 64 --use_fp16 True --num_channels 192 --attention_head_dim 64 --layers_per_block 3 \
    --variance_type learned_range --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine  \
    --rescale_learned_sigmas True  --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128 \
    --encoder_type bernoulli --latent_dim 64 --z_via_cross_att True  --cross_attention True
```
### Edges-64 and Handbags-64
```bash
DATSET=edges # or handbbags
mpiexec -n 2 python ddpm_train.py --data_dir data/edges2handbags/${DATASET}64_train --input_pertub 0.1 \
    --image_size 64 --use_fp16 True --num_channels 192 --attention_head_dim 64 --layers_per_block 3 \
    --variance_type learned_range --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine  \
    --rescale_learned_sigmas True  --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128  \
    --encoder_type bernoulli --latent_dim 256 --z_via_cross_att True --cross_attention True --one_cross_attention 2
```
Use `--resume_checkpoint` to resume `model` checkpoint (use `ema` checkpoints for generation).
Use `--finetune_path` to finetune (unconditional) model.
Train unconditional model by setting `--encoder_type none --latent_dim 0`.
Use `--microbatch 64` if resources are limited. Use `--one_cross_attention` parameter to train a lighter versions of DMZ.

### Finetuning CelebA-HQ DDPM from HuggingFace
Download DDPM checkpoint from [huggingaface](https://huggingface.co/google/ddpm-ema-celebahq-256).
Run
```bash
mpiexec -n 2 python ddpm_train.py --data_dir data/celebahq/celebahq256_train --input_pertub 0.1 \
    --image_size 256 --use_fp16 True --num_channels 128 --attention_head_dim 512 --layers_per_block 2 \
    --variance_type fixed_small --dropout 0.1 --diffusion_steps 1000 --noise_schedule linear  \
    --rescale_learned_sigmas True  --schedule_sampler loss-second-moment --lr 1e-4 --batch_size 128 \
    --encoder_type bernoulli --latent_dim 256 --z_via_cross_att True  --cross_attention True  \
    --finetune_path PRETRAINED_MODEL --resnet_time_scale_shift default --hugging_face_checkpoints_params True
```

## Image generation
Generate 10K images, measure NLL and get qualitative examples for CIFAR-10 by running
```bash
SAMPLES=10000
NLL_SAMPLES=1000
SAMPLER_NAME=random
mpiexec -n 1 python ddpm_eval.py --model_path CHECKPOINT \
        --train_data_dir data/cifar10/cifar10_train --test_data_dir data/cifar10/cifar10_test \
        --image_size 32 --use_fp16 False --num_channels 128 --attention_head_dim 32 --layers_per_block 3 \
        --variance_type learned_range --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine  \
        --rescale_learned_sigmas True  --batch_size 256 \
        --timestep_respacing 100 --num_samples SAMPLES --nll_num_samples NLL_SAMPLES \
        --encoder_type bernoulli --latent_dim 16 --z_via_cross_att True  --cross_attention True \
        --sampler_name ${SAMPLER_NAME}
```
Adjust parameters for other datasets based on previous section.

## Evaluate representations on downstream tasks
After extracting latent codes with `ddpm_eval.py`, evaluate them on downstream tasks by providing paths to `.pt` files with saved codes to run
```bash
python downstream_task_eval.py --latents_train LATENTS_TRAIN --latents_test LATENTS_TEST
```
Use `--auroc` flag for CelebA. Provide `--output_path` and use `--mlp` to save results and classifier weights, which can be passed to `ddpm_eval.py` via `--classifier_path` to perform image feature manipulation experiment.

## FID evaluation
Unpack npz files saved by `ddpm_eval.py` run with `num_samples=10000` using `unpack_npz.py` and [pytorch_fid](https://github.com/mseitzer/pytorch-fid).
Compare with the whole dataset.


## PixelSNAIL sampler
Train PixelSNAIL sampler on extracted latent codes, as in [VQVAE](https://github.com/rosinality/vq-vae-2-pytorch)

```bash
python pixelsnail_train.py --save_path PIXELSNAIL_SAVE_PATH  --latents_train LATENTS_TRAIN
```
Pass it to `ddpm_eval.py` using `--sampler_name pixelsnail --sampler_path PIXELSNAIL_SAVE_PATH` to generate images using PixelSNAIL sampler.

## Image-to-image with DMZ
First, train two DMZ models, one on edges-64 dataset, one on handbags-64 dataset (follow `datasets/edges2handbags64_npz.py` to setup the datasets).

### Learning $Z\to Z$ mapping (bridge)
After training two DMZ models with `ddpm_train.py` and extracting latent codes with `ddpm_eval.py`, train a mapping network with
```bash
python bridge_train.py --output_path BRIDGE_PATH \
    --latents_train_a INPUT_LATENTS_TRAIN --latents_test_a INPUT_LATENTS_TEST  \
    --latents_train_b TARGET_LATENTS_TRAIN --latents_test_b TARGET_LATENTS_TEST
```

### Running image-to-image
Generate images using edges latent codes and learned mapping.
```bash
mpiexec -n 1 python bridge_eval.py --model_path B_CHECKPOINT --bridge_path BRIDGE_PATH \
    --a_data_dir EDGES_DATA_PATH --latents_path HANDBAGS_LATENTS_TRAIN  \
    --image_size 64 --num_channels 192 --attention_head_dim 64 --layers_per_block 3 \
    --variance_type learned_range --learn_sigma True --diffusion_steps 1000 --noise_schedule cosine  \
    --rescale_learned_sigmas True  --batch_size 64 --encoder_type bernoulli --latent_dim 256 \
    --z_via_cross_att True --cross_attention True  --one_cross_attention 2 \
    --timestep_respacing 40 --num_samples 138567
```
Use `--start_idx` flag and `merge_npz.py` script if resources prevent you from running full evalutaion on almost 140K images.

### Evaluation
For running image-to-image evaluation metrics on generated by `bridge_eval.py` images saved in NZP_PATH, use [DDBM](https://github.com/alexzhou907/DDBM/) repository, torch-fidelity and [pytorch-fid](https://github.com/mseitzer/pytorch-fid/)
```bash
cd ../DDBM
python evaluations/evaluator.py ../diffusion/data/edges2handbags/handbags64_train.npz NPZ_PATH  --metric lpips

cd ../dmz
python unpack_npz.py --npz_path ${NPZ_PATH} --output_dir tmp
fidelity --input1 tmp  --isc
python -m pytorch_fid data/edges2handbags/handbags64_train tmp
rm -r tmp
```

## DDIM 
For training a DMZ/DDPM model that is compatible with DDIM sampling, set `--input_perturb 0` for training, and use `ddim_eval.py` and `--timestep_respacing ddim10` for evaluation.


## Mutual Information evaluation
To run the MI evaluation, extract intermediate $x_t$ from DMZ by running `ddpm_eval.py` with `--save_intermediate True` on DMZ trained on CIFAR-10.
This will create `latents_train.pt` and `latents_test.pt`, but also `intermediate_train_T100_19x50000x32x32x3.npz` and `intermediate_test_T100_19x50000x32x32x3.npz`.
Run
```bash
python mi_eval.py --latents_train latents_train.pt --latents_test latents_test.pt \
    --images_train intermediate_train_T100_19x50000x32x32x3.npz --images_test intermediate_test_T100_19x50000x32x32x3.npz \
python mi_eval.py --latents_train latents_train.pt --latents_test latents_test.pt \
    --data_dir_train data/cifar10/cifar_train --data_dir_test data/cifar10/cifar_test \
```
For more accurate MI eval, extract intermediate $x_t$ 20 times saving it as `intermediate_{train/test}_{0...19}_T100_19x50000x32x32x3.npz` and provide `intermediate_{train/test}_0_T100_19x50000x32x32x3.npz` to run the eval.
