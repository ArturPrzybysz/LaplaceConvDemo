import torch

from src import ROOT_DIR
from src.data_loader import KeypointDetectionDataset
from src.la_conv import LaplaceConvNet


def main():
    experiment_path = ROOT_DIR / "experiment_outputs"
    experiment_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KeypointDetectionDataset(device)

    hessian_structure = "diag"

    model = LaplaceConvNet(lr=3e-4, epochs=10, device=device, hessian_structure=hessian_structure)
    model.initialize_params()
    model.train(dataset.train_dataloader(), dataset.val_dataloader(), experiment_path)
    model.evaluate(dataset.test_dataset())

    sample = next(iter(dataset.test_dataset()))
    means, covariances = model.la(model)

    
if __name__ == '__main__':
    main()
