from pathlib import Path
from datasets import load_dataset
from typing import Tuple
from nflows.data import ocean
import torchvision as tv
import normflows as nf
import torch

class Unpack(object):
    """
    """

    def __call__(self, sample):
        image, y = sample['image'], sample['label']
        return image[0]

transform = tv.transforms.Compose(
    [
        # Unpack(),
        tv.transforms.ToTensor(),
        nf.utils.Scale(255.0 / 256.0),
        nf.utils.Jitter(1 / 256.0),
        tv.transforms.Resize((320, 320)),
    ]
)
from datasets import Image
from PIL import Image as PImage

from torchvision.datasets import flowers102

def transforms(samples):
    #
    samples["image"] = transform(samples["image"])
    samples["label"] = torch.Tensor(samples["label"])
    return samples

def get_flowers_dataset(batch_size=2):

    train_ds = load_dataset("nielsr/flowers-demo", split="train", keep_in_memory=False)
    # train_ds.set_transform(transforms)
    # https://github.com/huggingface/datasets/issues/5094
    # Deadlocks if num
    train_ds = train_ds.map(transforms,  batched=False, num_proc=1)
    train_ds.set_format("torch", columns=["image", "label"])
    train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )



    # test_ds = load_dataset("Guldeniz/flower_dataset", split="train").with_format("torch")
    # test_ds.set_transform(transform=transforms)
    # test_ds= test_ds.map(lambda x: x,  batched=True, batch_size=2)
    num_classes = 17
    return train_loader, train_loader, num_classes


def get_ocean_dataset(data_dir: Path, batch_size=2) -> Tuple[ torch.utils.data.DataLoader,  torch.utils.data.DataLoader, int]:
    train_ds = ocean.OceanData(root=data_dir, train=True, transform=transform)
    test_ds = ocean.OceanData(root=data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )

    test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
    num_classes = 1
    return train_loader, test_loader, num_classes

def get_flowers_102(batch_size:int=2):
    train_ds = flowers102.Flowers102(root="/mnt/vol_b/datasets", split="train", download=True, transform=transform)
    test_ds = flowers102.Flowers102(root="/mnt/vol_b/datasets", split="test", download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )

    test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, drop_last=True
        )
    num_classes = 1
    return train_loader, test_loader, num_classes


if __name__ =="__main__":
    # d = get_flowers_102()
    from tqdm import tqdm
    # load_dataset("Guldeniz/flower_dataset", split="train").save_to_disk("/mnt/vol_b/datasets/flower_dataset")
    train_ds,_,_  = get_flowers_102()
    for x in tqdm(train_ds):
        breakpoint()
        pass
        # print(len(x))