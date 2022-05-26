from pathlib import Path

import dill
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader

from src.visualisation import plot_sample


class RegularConvNet:
    def __init__(self, lr, epochs, device):
        self.name = f"Regular Conv. NN, epochs={epochs}, lr={lr}"
        self.lr = lr
        self.nn = None
        self.epochs = epochs
        self.device: str = device
        self.train_size = None

    def save(self, path: Path):
        torch.save(self.nn.state_dict(), path)

    def load(self, path: Path):
        self.initialize_params()
        with path.open("rb") as f:
            self.nn.load_state_dict(dill.load(f))

    def initialize_params(self):
        self.nn = SqueezeNet()
        self.nn = self.nn.double().to(self.device)

    def train(self, train_set: DataLoader, validation_set: DataLoader, experiment_path=None):
        self.train_size = len(train_set.dataset)

        avg_train_losses = []
        avg_valid_losses = []

        optimizer = torch.optim.Adam(self.nn.parameters(), self.lr)
        criterion = nn.MSELoss()

        for i in range(self.epochs):
            print_img = True
            print(f"Epoch {i} / {self.epochs}")
            train_losses = []
            valid_losses = []

            self.nn.train()
            for X, y in train_set:
                optimizer.zero_grad()
                pred = self.nn(X)
                loss1 = criterion(pred, y)

                train_losses.append(loss1.item())

                loss1.backward()
                optimizer.step()
                if print_img:
                    plot_sample(X.cpu().detach().numpy()[3], pred[3].cpu().detach().numpy(),
                                y[3].cpu().detach().numpy(),
                                f"train_epoch_{i}_a.png",
                                experiment_path / f"train_epoch_{i}_a.png")
                    plot_sample(X.cpu().detach().numpy()[5], pred[5].cpu().detach().numpy(),
                                y[5].cpu().detach().numpy(),
                                f"train_epoch_{i}_a.png",
                                experiment_path / f"train_epoch_{i}_b.png")
                    print_img = False
            self.nn.eval()  # prep model for evaluation
            with torch.no_grad():
                for X, y in validation_set:
                    output = self.nn(X)
                    loss2 = criterion(output, y)
                    valid_losses.append(loss2.item())
                    plot_sample(X.cpu().detach().numpy()[3], output[3].cpu().detach().numpy(),
                                y[3].cpu().detach().numpy(),
                                f"val_epoch_{i}_a.png",
                                experiment_path / f"val_epoch_{i}_a.png")
                    plot_sample(X.cpu().detach().numpy()[5], output[5].cpu().detach().numpy(),
                                y[5].cpu().detach().numpy(),
                                f"val_epoch_{i}_a.png",
                                experiment_path / f"val_epoch_{i}_b.png")

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)

                print("train_loss", train_loss)
                print("valid_loss", valid_loss)

                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

        with torch.no_grad():
            for X, y in validation_set:
                output = self.nn(X)
                loss = criterion(output, y)

                for idx in range(9):
                    plot_sample(X.cpu()[idx], output[idx].detach().cpu().numpy(), y.detach().cpu().numpy()[idx],
                                f"END at {loss.item()}")
                # plt.show()

        return avg_train_losses, avg_valid_losses

    def evaluate(self, test_set: DataLoader):
        super().evaluate(test_set)

        self.nn.eval()
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []
            for X, y in test_set:
                y_pred = self.nn(X.expand(-1, 3, -1, -1))
                plot_sample(X.cpu()[2], y_pred[2], y[2])
                y_pred_list.append(y_pred.cpu().numpy())
                y_true_list.append(y.cpu().numpy())

        # y_true = np.hstack(y_true_list)
        # y_pred = np.vstack(y_pred_list)


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(self, target_count: int = 30) -> None:
        super().__init__()
        self.target_count = target_count
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
            nn.Conv2d(512, 64, 1)
        )

        self.regression = nn.Sequential(
            nn.Linear(1600, 256), nn.Linear(256, self.target_count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regression(x)
        return x
