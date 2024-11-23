# data_preparation.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_data(batch_size=64):
    # Định nghĩa các phép biến đổi dữ liệu
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển đổi ảnh thành tensor
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa dữ liệu
    ])

    # Tải dữ liệu MNIST
    full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Chia full_train_dataset thành train_dataset và val_dataset với tỷ lệ 90%-10%
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Tạo DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
