
import torch
from torch import nn
import torch.nn.functional as F
from dataset import CustomDataset
from torchvision.transforms import Resize, InterpolationMode


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # thread 0
        self.conv0_0 = nn.Conv2d(3, 5, kernel_size=7, stride=1, padding=3)
        self.conv0_1 = nn.Conv2d(5, 12, kernel_size=5, stride=1, padding=2)
        self.conv0_2 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.conv0_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv0_4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv0_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv0_6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv0_7 = nn.Conv2d(128, 32, kernel_size=7, stride=1, padding=3)
        self.conv0_8 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv0_9 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)

        # thread 1
        self.conv1_0 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)
        self.conv1_1 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3)
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv1_4 = nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv1_5 = nn.Conv2d(16, 4, kernel_size=7, stride=1, padding=3)

        # thread 3
        self.conv2_0 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.conv2_1 = nn.Conv2d(8, 24, kernel_size=11, stride=1, padding=5)
        self.conv2_2 = nn.Conv2d(24, 32, kernel_size=11, stride=1, padding=5)
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=11, stride=1, padding=5)
        self.conv2_4 = nn.Conv2d(32, 64, kernel_size=11, stride=1, padding=5)
        self.conv2_5 = nn.Conv2d(64, 128, kernel_size=11, stride=1, padding=5)
        self.conv2_6 = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5)
        self.conv2_7 = nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.conv2_8 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_9 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        # final 
        self.conv3_0 = nn.Conv2d(28, 32, kernel_size=7, stride=1, padding=3)
        self.conv3_1 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(8, 3, kernel_size=1, stride=1)
        
        self.activation_0 = nn.ReLU()
        self.activation_1 = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = self.conv0_0(x)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_1(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_2(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.pool(x_0)
        x_0 = self.conv0_3(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_4(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_5(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_6(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.pool(x_0)
        x_0 = self.conv0_7(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.conv0_8(x_0)
        x_0 = self.activation_0(x_0)
        x_0 = self.pool(x_0)
        x_0 = self.conv0_9(x_0)
        x_0 = Resize((360, 360), InterpolationMode.NEAREST)(x_0)

        x_1 = self.conv2_0(x)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_1(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_2(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.pool(x_1)
        x_1 = self.conv2_3(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_4(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_5(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_6(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.pool(x_1)
        x_1 = self.conv2_7(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.conv2_8(x_1)
        x_1 = self.activation_0(x_1)
        x_1 = self.pool(x_1)
        x_1 = self.conv2_9(x_1)
        x_1 = Resize((360, 360), InterpolationMode.NEAREST)(x_1)

        x_2 = self.conv1_0(x)
        x_2 = self.activation_0(x_2)
        x_2 = self.conv1_1(x_2)
        x_2 = self.activation_0(x_2)
        x_2 = self.conv1_2(x_2)
        x_2 = self.activation_0(x_2)
        x_2 = self.conv1_3(x_2)
        x_2 = self.activation_0(x_2)
        x_2 = self.conv1_4(x_2)
        x_2 = self.activation_0(x_2)
        x_2 = self.conv1_5(x_2)

        x_3 = torch.concat((x_0, x_1, x_2), dim=1)
        x_3 = self.conv3_0(x_3)
        x_3 = self.activation_1(x_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.activation_1(x_3)
        x_3 = self.conv3_2(x_3)
        x_3 = self.activation_1(x_3)
        x_3 = self.conv3_3(x_3)

        return x_3

if __name__ == "__main__":
    ds = CustomDataset()
    model = Model()

    print(model(ds[1][0].unsqueeze(0)).shape)