import os
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn
import pdb

from src import config
from src.loop_toys import loop
from src.synthetic_functions import (
    generate_objective_from_gp_post,
    compute_rewards,
    get_lengthscale_hyperprior,
)


class Objective_Functions():
       
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
        self.z = torch.Tensor([0.0, 0.0])
        self.weight = torch.Tensor([0.5, 0.5])

    def aug_tch(self, f_x):
        tch = torch.max(self.weight * (f_x - (self.z - torch.abs(self.z * self.e))), axis = 1)[0]
        ws = torch.sum(self.weight * f_x, axis = 1)
        
        return (tch + self.p * ws)

def objective(x):
    obj_func = Objective_Functions()
    scalar = Scalar()
    f_x = obj_func.evaluate(x)
    tch_f_x = scalar.aug_tch(f_x).squeeze().unsqueeze(0)
    return tch_f_x
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run optimization of synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")
    parser.add_argument("-cd", "--config_data", type=str, help="Path to data config file.")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Translate config dictionary.
    cfg = config.insert(cfg, config.insertion_config)

    with open(args.config_data, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)


    directory = cfg["out_dir"]
    if not os.path.exists(directory):
        os.makedirs(directory)

    for dim in cfg_data["dimensions"]:
        print(f"\nDimension {dim}:")

        x_list = []
        fx_list = []
        # calls_list = []

        for index_objective in range(cfg_data["num_objectives"]):

            cfg_dim = config.evaluate(
                cfg,
                dim_search_space=dim,
                factor_lengthscale=cfg_data["factor_lengthscale"],
                factor_N_max=5,
                hypers=None)
            
            x0 = np.zeros(dim)
            x0[range(0, dim, 2)] = 0.6
            x0[range(1, dim, 2)] = -.3
            x0 = torch.Tensor(x0)

            x_out, fx_out, calls_in_iteration = loop(
                params_init= x0,
                max_iterations=cfg_dim["max_iterations"],
                max_objective_calls=cfg_dim["max_objective_calls"],
                objective=objective,
                optimizer_config=cfg_dim["optimizer_config"],
                verbose=True)
            # pdb.set_trace()

            # print(f"Optimizer's max reward: {max(rewards)}")
            x_list.append(x_out)
            fx_list.append(fx_out)
            # calls_list.append(calls_in_iteration)
            # pdb.set_trace();

            print(f"Save parameters, objective calls and rewards (function values) at {directory}.")
            # n_iteration + 1 => the first one is init
            np.save(os.path.join(directory, "x_" + str(dim) + "_BO"), x_list)
            # dict{n_dim: list(n_obj, n_iteration + 1, Tensor[1, n_dim])}
            np.save(os.path.join(directory, "fx_" + str(dim) + "_BO"), fx_list[0])
            # np.save(os.path.join(directory, "calls_"+dim), calls_list)
            # dict{n_dim: list(1, n_iteration + 1)}
