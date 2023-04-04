from typing import Dict, Callable, Optional, List

import numpy as np
import torch
import gpytorch
import botorch

from src.environment_api import EnvironmentObjective
from src.model import ExactGPSEModel

class VanillaBayesianOptimization(AbstractOptimizer):
    """Optimizer class for vanilla Bayesian optimization.

    Vanilla stands for the usage of a classic acquisition function like
    expected improvement.

    Atrributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        Model: Gaussian process model.
        model_config: Configuration dictionary for model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        acquisition_function: BoTorch acquisition function.
        acqf_config: Configuration dictionary acquisition function.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
        Model,
        model_config: Dict,
        hyperparameter_config: Optional[Dict],
        acquisition_function,
        acqf_config: Dict,
        optimize_acqf: Callable,
        optimize_acqf_config: Dict[str, torch.Tensor],
        generate_initial_data=Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ],
        verbose=True,
    ):
        """Inits the vanilla BO optimizer."""
        super(VanillaBayesianOptimization, self).__init__(params_init, objective)

        # Parameter initialization.
        self.params_history_list = [self.params.clone()]
        self.D = self.params.shape[-1]

        # Initialization of training data.
        if generate_initial_data is None:
            train_x_init, train_y_init = torch.empty(0, self.D), torch.empty(0, 1)
        else:
            train_x_init, train_y_init = generate_initial_data(self.objective)

        # Add initialization parameter to training data.
        train_x_init = torch.cat([train_x_init, self.params])
        train_y_init = torch.cat(
            [train_y_init, self.objective(self.params).reshape(-1, 1)]
        )

        # Model initialization and optional hyperparameter settings.
        self.model = Model(train_x_init, train_y_init, **model_config)
        if hyperparameter_config["hypers"]:
            self.model.initialize(**hyperparameter_config["hypers"])
        if hyperparameter_config["no_noise_optimization"]:
            # Switch off the optimization of the noise parameter.
            self.model.likelihood.noise_covar.raw_noise.requires_grad = False

        self.optimize_hyperparamters = hyperparameter_config["optimize_hyperparameters"]

        # Acquistion function and its optimization properties.
        self.acquisition_function = acquisition_function
        self.acqf_config = acqf_config
        self.optimize_acqf = optimize_acqf
        self.optimize_acqf_config = optimize_acqf_config

        self.verbose = verbose

    def step(self) -> None:
        # Optionally optimize hyperparameters.
        if self.optimize_hyperparamters and self.model.train_targets.shape[0] > 20:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.model.likelihood, self.model
            )
            botorch.fit.fit_gpytorch_model(mll)

        # Optionally update best_f for acquistion function.
        if "best_f" in self.acqf_config.keys():
            self.acqf_config["best_f"] = self.model.train_targets.max()

        # Optimize acquistion function and get new observation.
        new_x = self.optimize_acqf(
            self.acquisition_function(self.model, **self.acqf_config),
            **self.optimize_acqf_config,
        )
        new_y = self.objective(new_x)
        self.params = new_x.clone()

        # Update training points.
        train_x = torch.cat([self.model.train_inputs[0], new_x])
        train_y = torch.cat([self.model.train_targets, new_y])
        self.model.set_train_data(inputs=train_x, targets=train_y, strict=False)

        self.params_history_list.append(self.params)

        if self.verbose:
            posterior = self.model.posterior(self.params)
            print(
                f"Parameter {self.params.numpy()} with mean {posterior.mvn.mean.item(): .2f} and variance {posterior.mvn.variance.item(): .2f} of the posterior of the GP model."
            )
            print(
                f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach().numpy()}, outputscale: {self.model.covar_module.outputscale.detach().numpy(): .2f},  noise {self.model.likelihood.noise.detach().numpy()}"
            )