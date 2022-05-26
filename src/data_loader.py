import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from albumentations.pytorch import ToTensorV2

from . import DATA_DIR
import albumentations as A
import cv2


class KeypointDetectionDataset:
    def __init__(self, device, augment_train=True, data_path=DATA_DIR / "facial-keypoints-detection", batch_size=32):
        self.batch_size = batch_size
        images, keypoints = self.__load_dataset(data_path)
        X_pretrain, X_test, y_pretrain, y_test = train_test_split(images, keypoints, test_size=0.1, random_state=21)

        imputer = KNNImputer()
        y_pretrain = imputer.fit_transform(y_pretrain.reshape(-1, 30)).reshape(-1, 30, 1)
        y_test = imputer.transform(y_test.reshape(-1, 30)).reshape(-1, 30, 1)

        X_train, X_val, y_train, y_val = train_test_split(X_pretrain, y_pretrain, train_size=0.99, random_state=37)

        self.train_dataset = DataSource(X_train, y_train, augment=augment_train, device=device)
        self.test_dataset = DataSource(X_test, y_test, augment=False, device=device)
        self.val_dataset = DataSource(X_val, y_val, augment=False, device=device)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, drop_last=True, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, drop_last=True, batch_size=self.batch_size)

    def __load_dataset(self, dir_path):
        train_df = pd.read_csv(dir_path / "training.csv")

        images_train = self.__load_images(train_df)
        train_keypoints = self.__load_keypoints(train_df)

        return images_train, train_keypoints

    def __load_images(self, image_data):
        images = []
        for idx, sample in image_data.iterrows():
            image = np.array(sample['Image'].split(' '), dtype=int)
            image = np.reshape(image, (96, 96, 1))
            images.append(image)
        images = np.array(images) / 255.
        return images

    def __load_keypoints(self, keypoint_data):
        keypoint_data = keypoint_data.drop('Image', axis=1)
        keypoint_features = []
        for idx, sample_keypoints in keypoint_data.iterrows():
            keypoint_features.append(sample_keypoints)
        keypoint_features = np.array(keypoint_features, dtype='float')
        # keypoint_features = np.hstack([keypoint_features[:, 0::2], keypoint_features[:, 1::2]])

        return keypoint_features.reshape((-1, 15, 2))


class DataSource(Dataset):
    def __init__(self, images, keypoints, augment, device):
        self.images = images
        self.keypoints = keypoints
        self.augment = augment
        self.device = device
        self.transform = A.Compose([A.Resize(width=96, height=96),
                                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=65, p=1,
                                                       border_mode=cv2.BORDER_CONSTANT),
                                    A.Blur(p=0.3, blur_limit=3),
                                    ToTensorV2(),
                                    ],

                                   keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

        self.non_modifying_transform = A.Compose([A.Resize(width=96, height=96), ToTensorV2()],
                                                 keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    def __len__(self):
        return self.keypoints.shape[0]

    def __getitem__(self, index):
        X = self.images[index]
        y = self.keypoints[index]

        if self.augment:
            transformed = self.transform(image=X, keypoints=y.reshape((15, 2)))
            X = transformed["image"].expand(3, -1, -1)
            y = torch.Tensor(transformed["keypoints"]).reshape(30).double().to(self.device)
        else:
            transformed = self.non_modifying_transform(image=X, keypoints=y.reshape((15, 2)))
            X = transformed["image"].expand(3, -1, -1)
            y = torch.Tensor(transformed["keypoints"]).reshape(30).double().to(self.device)
        X = X.double().to(self.device)
        return X, y

