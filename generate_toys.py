import os
import yaml
import argparse
import pdb
import numpy as np
from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List
from torch import Tensor

import torch

from src.synthetic_functions import (
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
)

class cybenko():
    def __init__(self, n_dim = 1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.nadir_point = [1, 1]
       
    def evaluate(self, x):        
        n = x.shape[1]
        
        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs

class Scalar():
    def __init__(self):
        self.p = 0.001
        self.e = 0.1

    def aug_tch(self, f, ref, z):
        tch = torch.argmax(ref * (f - (z - torch.abs(z * self.e))), axis = 1)
        ws = torch.sum(ref * f, axis = 1)
        return (tch + self.p * ws).unsqueeze(1)

def generate_training_samples(
    num_objectives: int, dim: int, num_samples: int
) -> Tuple[List[Tensor], List[Tensor]]:
    """Generate training samples for `num_objectives` objectives.

    Args:
        num_objectives: Number of objectives.
        dim: Dimension of parameter space/sample grid.
        num_samples: Number of grid samples.

    Returns:
        List of trainings features and targets.
    """

    pb = cybenko()
    train_x = []
    train_y = []
    for _ in range(num_objectives):
        # Random tensor with shape (num_samples, dim) follow Uniform distribution within range [-1, 1]
        x = torch.Tensor.uniform_(torch.empty(num_samples, dim), -1, 1)
        truth = pb.evaluate(x)
        weight = torch.Tensor([0.5, 0.5])

        ideal_vector = torch.Tensor([0.0, 0.0])

        scalar = Scalar()
        y = scalar.aug_tch(truth, weight, ideal_vector).squeeze()
        train_x.append(x)
        train_y.append(y)
    return train_x, train_y

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate data for synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)

    lengthscales_dict = {}
    train_x_dict = {}
    train_y_dict = {}
    # z = torch.zeros(n_obj).to(device)


    for dim in cfg_data["dimensions"]:
        print(f"Data for function with {dim}-dimensional domain.")
        l = get_lengthscales(dim, factor_hennig)
        m = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]),
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]),
        )
        lengthscale = m.sample((1, dim))
        train_x, train_y = generate_training_samples(
            num_objectives=cfg_data["num_objectives"],
            dim=dim,
            num_samples=cfg_data["num_samples"],
        )
        train_x_dict[dim] = train_x
        train_y_dict[dim] = train_y
        lengthscales_dict[dim] = lengthscale


    print("Compute maxima and argmax of synthetic functions.")
    f_max_dict, argmax_dict = get_maxima_objectives(
        lengthscales=lengthscales_dict,
        noise_variance=cfg_data["noise_variance"],
        train_x=train_x_dict,
        train_y=train_y_dict,
        n_max=cfg_data["n_max"],
    )
    # pdb.set_trace()

    if not os.path.exists(cfg_data["out_dir"]):
        os.mkdir(cfg_data["out_dir"])

    path = cfg_data["out_dir"]
    print(f"Save data at {path}.")
    torch.save(train_x_dict, os.path.join(path, "train_x.pt"))
    torch.save(train_y_dict, os.path.join(path, "train_y.pt"))
    torch.save(lengthscales_dict, os.path.join(path, "lengthscales.pt"))
    torch.save(f_max_dict, os.path.join(path, "f_max.pt"))
    torch.save(argmax_dict, os.path.join(path, "argmax.pt"))
