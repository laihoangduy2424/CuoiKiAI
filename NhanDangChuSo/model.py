# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Lớp tích chập 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)

        # Lớp tích chập 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        # Lớp tích chập 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        # Lớp pooling 2D
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tính toán kích thước đầu ra cho fully connected layer
        self.fc_input_size = self.calculate_fc_input_size()

        # Lớp fully connected
        self.fc1 = nn.Linear(self.fc_input_size, 256)  # Cập nhật kích thước
        self.fc2 = nn.Linear(256, 10)  # 10 lớp cho các chữ số từ 0 đến 9

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Qua lớp conv1 và pooling
        x = self.pool(F.relu(self.conv2(x)))  # Qua lớp conv2 và pooling
        x = F.relu(self.conv3(x))  # Qua lớp conv3 mà không có pooling

        # Flatten
        x = torch.flatten(x, 1)  # Flatten từ chiều thứ 1 trở đi (giữ lại batch size)

        x = F.relu(self.fc1(x))  # Qua lớp fully connected đầu tiên
        x = self.fc2(x)  # Qua lớp fully connected cuối cùng
        return x

    def calculate_fc_input_size(self):
        # Tạo một tensor giả để tính toán kích thước đầu ra sau các lớp conv và pool
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # Kích thước đầu vào cho MNIST
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            return x.numel()  # Số lượng phần tử trong tensor


# Khởi tạo mô hình
model = SimpleCNN()
print(model)
