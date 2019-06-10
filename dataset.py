from typing import Union, List, Callable

import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# fixme: NOT TESTED
class DogBreedDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'data/train',
                 image_file_extension: str = '.jpg',
                 labels: Union[str, None] = 'data/labels.csv',
                 transformation: Union[List[Callable[[torch.FloatTensor], torch.FloatTensor]], None] = None,
                 is_test: Union[bool, None] = False,
                 test_size: Union[float, None] = 0.4,
                 seed: int = 666):
        """
        Dataset designed to work in conjunction with `albumentations` library
        https://github.com/albu/albumentations/blob/master/notebooks/migrating_from_torchvision_to_albumentations.ipynb

        :param data_dir: Directory with images
        :param image_file_extension: For example '.png'
        :param labels:
            Path to csv file with labels.
            If this parameter is None, dataset will return all data without labels
        :param transformation: callable object for data transformation/augmentation
        :param is_test: whether to return test part of test/train split
        :param test_size: size of test dataset (for example 0.25)
        :param seed: random seed for test/train split reproducibility.
        """
        self.data_dir = data_dir
        self.is_inference = labels is None  # inference -- for kaggle submission
        self.is_test = is_test  # for model training and validation
        self.transformation = transformation
        self._random_state = np.random.RandomState(seed=seed)

        images_df = pd.DataFrame({'img': [file_name for file_name in os.listdir(data_dir)]})
        images_df['id'] = images_df['img'].str.replace(image_file_extension, '')

        images_df.dropna(inplace=True)

        if self.is_inference:
            images_df['breed'] = np.nan
            self._data = images_df
        else:
            labels_df = pd.read_csv(labels, index_col=False).dropna()
            self._data = images_df.merge(labels_df, how='inner', left_on='id', right_on='id')

        self._data.dropna(inplace=True)
        self._data.reset_index(drop=True, inplace=True)

        if self.is_inference:
            pass
        else:
            train_idxs, test_idxs = train_test_split(
                self._data.index, test_size=test_size, random_state=self._random_state)

            if is_test:
                self._data = self._data.loc[test_idxs, :]
            else:
                self._data = self._data.loc[train_idxs, :]

    def __len__(self):
        return len(self._data.index)

    def __getitem__(self, item):
        image = cv2.cvtColor(
            cv2.imread(os.path.join(self.data_dir, self._data.iloc[item, 'img'])),
            cv2.COLOR_BGR2RGB)

        if not self.is_inference:
            label = self._data.iloc[item, 'breed']

        if self.transformation is not None:
            augmented = self.transformation(image=image)
            image = augmented['image']

        return image, label
