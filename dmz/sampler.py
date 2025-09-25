from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch

from .pixelsnail import PixelSNAIL


class Sampler(ABC):
    def __init__(self):
        """
        Initializes the Sampler with data.

        :param data: The data to sample from.
        """
        pass

    @abstractmethod
    def sample(self, N, device):
        """
        Abstract method that should be implemented by subclasses to
        sample data in a specific way.
        """
        pass


def extend_bernoulli_latent(x):
    x = x.unsqueeze(-1)
    return torch.cat([1 - x, x], dim=-1).reshape(x.shape[0], -1)


class PixelSNAILSampler(Sampler):
    def __init__(self, encoder_type, sampler_path, device):
        super(PixelSNAILSampler, self).__init__()
        self.encoder_type = encoder_type
        loaded = torch.load(sampler_path, map_location="cpu", weights_only=False)
        state_dict = loaded["state_dict"]
        model_args = loaded["model_args"]
        sampler = PixelSNAIL(**model_args)
        sampler.load_state_dict(state_dict)
        sampler.eval()
        sampler = sampler.to(device)
        self.sampler = sampler
        self.shape = model_args["shape"]

    def sample(self, N, device):
        samples = torch.zeros([N, *self.shape], dtype=torch.long, device=device)
        cache = {}
        temperature = 1
        with torch.no_grad():
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    out, cache = self.sampler(samples[:, : i + 1, :], cache=cache)
                    prob = torch.softmax(out[:, :, i, j] / temperature, 1)
                    sample = torch.multinomial(prob, 1).squeeze(-1)
                    samples[:, i, j] = sample

        samples = samples.reshape(N, -1)
        return extend_bernoulli_latent(samples) if self.encoder_type == "bernoulli" else samples


class RandomSampler(Sampler):
    def __init__(self, encoder_type, latent_dim, val_max):
        super(RandomSampler, self).__init__()
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.val_max = val_max

    def sample(self, N, device):
        if self.encoder_type == "bernoulli":
            samples = torch.bernoulli(0.5 * torch.ones(N, *self.latent_dim)).to(device)
            return extend_bernoulli_latent(samples[:, : samples.shape[1] // 2])
        elif self.encoder_type == "normal":
            samples = torch.rand(N, *self.latent_dim).to(device)
            return samples
        else:
            return torch.randint(low=0, high=self.val_max, size=(N, *self.latent_dim)).to(device).long()


class NormalSampler(Sampler):
    def __init__(self, latents):
        super(NormalSampler, self).__init__()
        self.mu = latents.mean(0)
        self.std = latents.std(0)
        self.latent_dim = latents.shape[-1]

    def sample(self, N, device):
        samples = torch.rand(N, self.latent_dim).to(device)
        print(self.mu.shape, samples.shape, self.std.shape)
        samples = self.mu.to(device) + samples * self.std.to(device)
        return samples


class ConstSampler(Sampler):
    def __init__(self, const_val, latent_dim):
        super(ConstSampler, self).__init__()
        self.const_val = const_val
        self.latent_dim = latent_dim

    def sample(self, N, device):
        return self.const_val.to(device) * torch.ones(N, *self.latent_dim).to(device)


class DatasetSampler(Sampler):
    def __init__(self, latents):
        super(DatasetSampler, self).__init__()
        self.latents = latents.detach().cpu()

    def sample(self, N, device):
        idx = torch.randint(low=0, high=len(self.latents), size=(N,))
        return self.latents[idx].to(device)

class EmpiricalSampler(Sampler):
    def __init__(self, latents):
        super(EmpiricalSampler, self).__init__()
        self.p = latents.detach().cpu().mean(0)

    def sample(self, N, device):
        samples = torch.bernoulli(self.p.repeat(N, 1)).to(device)
        return extend_bernoulli_latent(samples[:, : samples.shape[1] // 2])

