from pathlib import Path
from typing import List, Tuple

import torch
from laplace import Laplace
from torch.utils.data import DataLoader

from src.regular_conv import RegularConvNet
from src.visualisation import plot_sample_LA


class LaplaceConvNet:
    def __init__(self, lr, epochs, device, link_approx="mc", hessian_structure="diag"):
        self.la: Laplace = None
        self.name = f"Laplace Conv. NN, epochs={epochs}, lr={lr}"
        self.lr = lr
        self.link_approx = link_approx
        self.regular_convnet: RegularConvNet = None
        self.epochs = epochs
        self.device: str = device
        self.train_size = None
        self.hessian_structure = hessian_structure

    @staticmethod
    def from_trained_conv_net(model: RegularConvNet, link_approx="mc") -> "LaplaceConvNet":
        model.epochs = 0
        laplace_convnet = LaplaceConvNet(model.lr, model.epochs, model.device, link_approx)
        laplace_convnet.regular_convnet = model
        return laplace_convnet

    def initialize_params(self):
        """Set parameters to initial values, so that experiments are independent"""
        self.regular_convnet = RegularConvNet(self.lr, self.epochs, self.device)
        self.regular_convnet.initialize_params()

    def train_regular_network_and_LA(self, train_set: DataLoader, validation_set: DataLoader, experiment_path: Path) -> \
            Tuple[
                List, List]:
        self.train_size = len(train_set.dataset)

        avg_train_losses, avg_valid_losses = self.regular_convnet.train(train_set, validation_set, experiment_path)
        self.regular_convnet.nn.train()
        self.la = Laplace(self.regular_convnet.nn, 'regression', hessian_structure=self.hessian_structure)
        self.la.fit(train_set)
        self.la.optimize_prior_precision(val_loader=validation_set)

        return avg_train_losses, avg_valid_losses

    def train_LA(self, train_set: DataLoader, validation_set: DataLoader):
        self.la = Laplace(self.regular_convnet.nn, 'regression', hessian_structure=self.hessian_structure)
        self.la.fit(train_set)
        self.la.optimize_prior_precision(val_loader=validation_set)

    def evaluate(self, test_set: DataLoader, first_batch_only=True):
        means, covariances, y_true = [], [], []
        for X, y in test_set:
            pred = self.la(X, link_approx=self.link_approx)
            mean, covariance = pred
            means.append(mean)
            covariance_diag = torch.diagonal(covariance, dim1=1, dim2=2) + self.la.sigma_noise
            covariances.append(covariance_diag)
            for i in range(y.shape[0]):
                plot_sample_LA(X[i], mean[i], covariance_diag[i], self.la.sigma_noise)
            if first_batch_only:
                break  # shortcut to plot only 1 batch.
