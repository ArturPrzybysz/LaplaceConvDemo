from matplotlib import pyplot as plt
import numpy as np


def _compute_gaussian2d(x, y, var_x, var_y, from_=0, to_=96):
    m = np.array([x, y])
    var_x = var_x.cpu().detach().numpy()
    var_y = var_y.cpu().detach().numpy()
    cov = np.array([[var_x, 0], [0., var_y]])
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    x = np.linspace(from_, to_, num=500)
    y = np.linspace(from_, to_, num=500)
    X, Y = np.meshgrid(x, y)
    coe = 1.0 / ((2 * np.pi) ** 2 * cov_det) ** 0.5
    Z = coe * np.e ** (-0.5 * (
            cov_inv[0, 0] * (X - m[0]) ** 2 + (cov_inv[0, 1] + cov_inv[1, 0]) * (X - m[0]) * (Y - m[1]) + cov_inv[
        1, 1] * (Y - m[1]) ** 2))
    return X, Y, Z


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


def plot_sample_LA(image, keypoint, keypoint_variance, sigma_noise):
    image = image.mean(axis=0)
    keypoints_x, keypoints_y = keypoint[0::2], keypoint[1::2]
    keypoint_variance = keypoint_variance + sigma_noise
    var_x, var_y = keypoint_variance[0::2], keypoint_variance[1::2]

    mycmap = transparent_cmap(plt.cm.Reds)

    X, Y, Z = None, None, None
    for i in range(len(var_x)):
        x = keypoints_x[i]
        y = keypoints_y[i]
        vx = var_x[i]
        vy = var_y[i]
        X, Y, z = _compute_gaussian2d(x, y, vx, vy)
        if Z is None:
            Z = z
        else:
            Z += z
    plt.imshow(image, cmap='gray')
    plt.contourf(X, Y, Z, alpha=0.5, cmap=mycmap)

    plt.show()


def plot_sample(image, keypoint, target_keypoints, title="", img_path=None):
    if len(image.shape) == 4:
        image = image.mean(axis=(0, 1))
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = image.mean(axis=0)
    else:
        raise NotImplementedError()
    plt.imshow(image, cmap='gray')
    plt.scatter(target_keypoints[0::2], target_keypoints[1::2], marker='o', s=20, c='red')
    plt.scatter(keypoint[0::2], keypoint[1::2], marker='x', s=20)
    plt.title(title)
    if img_path is None:
        plt.show()
    else:
        plt.savefig(img_path)
        plt.clf()


def plot_sample_pred_scatter(image, keypoint_predictions, target_keypoints, title="", img_path=None):
    if len(image.shape) == 4:
        image = image.mean(axis=(0, 1))
    elif len(image.shape) == 3 and image.shape[0] == 3:
        image = image.mean(axis=0)
    else:
        raise NotImplementedError()
    plt.imshow(image, cmap='gray')
    plt.scatter(target_keypoints[0::2], target_keypoints[1::2], marker='o', s=20, c='red')

    variational_samples = keypoint_predictions.shape[0]

    for i in range(variational_samples):
        plt.scatter(keypoint_predictions[i, 0::2], keypoint_predictions[i, 1::2], marker='x', s=1, c='blue')

    plt.title(title)
    if img_path is None:
        plt.show()
    else:
        plt.savefig(img_path)
        plt.clf()
