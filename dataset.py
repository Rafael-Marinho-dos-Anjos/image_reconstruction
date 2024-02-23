
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from os import listdir
from torch import float32


class CustomDataset(Dataset):
    def __init__(self):
        self.images = listdir("data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        gt_path = path[: path.find("_")] + path[path.find("."):]

        X = read_image("data\\" + path).to(dtype=float32) / 255
        Y = read_image("gt\\" + gt_path).to(dtype=float32) / 255

        return X, Y
