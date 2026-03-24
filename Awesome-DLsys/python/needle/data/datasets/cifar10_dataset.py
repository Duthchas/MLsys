import os
import pickle
from typing import Optional, List
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[float] = 0.5,
        transforms: Optional[List] = None,
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images stored as H x W x C for transform compatibility
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms=transforms)
        self.p = p

        batch_files = (
            [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        )
        images: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for batch in batch_files:
            file_path = os.path.join(base_folder, batch)
            with open(file_path, "rb") as fh:
                batch_dict = pickle.load(fh, encoding="latin1")

            data = batch_dict["data"].reshape(-1, 3, 32, 32)
            # store as HWC to keep compatibility with image transforms
            images.append(np.transpose(data, (0, 2, 3, 1)))
            labels.append(np.array(batch_dict["labels"], dtype=np.int64))

        self.X = np.concatenate(images, axis=0).astype(np.float32) / 255.0
        self.y = np.concatenate(labels, axis=0)
        ### END YOUR SOLUTION

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        img = self.apply_transforms(img)
        return np.transpose(img, (2, 0, 1)).astype(np.float32, copy=False)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, (int, np.integer)):
            return self._process_image(self.X[int(index)]), int(self.y[int(index)])

        if isinstance(index, slice):
            indices = np.arange(len(self))[index]
        else:
            indices = np.array(index, dtype=np.int64)

        imgs = [self._process_image(self.X[i]) for i in indices]
        labels = self.y[indices].astype(np.int64, copy=False)
        return np.stack(imgs, axis=0), labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
