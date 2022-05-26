import torch

from src import ROOT_DIR
from src.data_loader import KeypointDetectionDataset
from src.la_conv import LaplaceConvNet
from src.regular_conv import RegularConvNet
from src.visualisation import plot_sample_LA


def main():
    experiment_path = ROOT_DIR / "experiment_outputs"
    experiment_path.mkdir(parents=True, exist_ok=True)

    device = "cpu"  # switch to "cuda" if you have GPU available

    dataset = KeypointDetectionDataset(device)

    hessian_structure = "diag"

    use_pretrained_model = True

    if use_pretrained_model:
        regular_conv_net = RegularConvNet(3e-4, epochs=0, device=device)
        regular_conv_net.load(ROOT_DIR / "state_dict.dill")
        la_model = LaplaceConvNet.from_trained_conv_net(regular_conv_net)
        la_model.train_LA(dataset.train_dataloader(), dataset.val_dataloader())
    else:
        la_model = LaplaceConvNet(lr=1e-4, epochs=250, device=device, hessian_structure=hessian_structure)
        la_model.initialize_params()
        la_model.train_regular_network_and_LA(dataset.train_dataloader(), dataset.val_dataloader(), experiment_path)

    # Visualise results and plot covariance!
    la_model.evaluate(dataset.test_dataloader())

    means, covariances, y_true = [], [], []
    for X, y in dataset.test_dataloader():
        pred = la_model.la(X)
        mean, covariance = pred
        means.append(mean)
        covariance_diag = torch.diagonal(covariance, dim1=1, dim2=2) + la_model.la.sigma_noise

        covariances.append(covariance_diag)
        batch_size = y.shape[0]
        for i in range(batch_size):
            print(covariances[i])
            assert torch.all(covariance_diag[0] == covariance_diag[i])  # The value are all the same!
            plot_sample_LA(X[i], mean[i], covariance_diag[i], la_model.la.sigma_noise)
        break  # shortcut to plot only 1 batch.


if __name__ == '__main__':
    main()
