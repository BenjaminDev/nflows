from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
import torch
from pathlib import Path
from PIL.Image import Image as PImage
from PIL import Image
from torchvision.datasets.folder import pil_loader

class OceanData(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        # download: bool = False,
    ) -> None:

        # super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set


        self.transform = transform
        self.target_transform = target_transform



        self.data: Path = list(Path(root).glob("*.png"))
        self.targets = [1]*len(self.data)


    def __getitem__(self, index: int) -> Tuple[PImage, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # breakpoint()
        img, target = Image.open(self.data[index]), 1 #self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # breakpoint()
        return img, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    od = OceanData(root="/mnt/vol_b/datasets/oceans_small_320_320")
    print(len(od))

