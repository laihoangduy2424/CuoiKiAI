# evaluate.py

import torch
from data_preparation import load_data
from model import SimpleCNN

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải dữ liệu
train_loader, val_loader, test_loader = load_data()  # Cập nhật để nhận cả val_loader

# Khởi tạo mô hình
model = SimpleCNN().to(device)

# Tải trọng số đã lưu (nếu có)
model.load_state_dict(torch.load('model.pth', map_location=device))  # Sử dụng map_location cho đúng thiết bị

# Đánh giá mô hình
model.eval()  # Chuyển mô hình sang chế độ đánh giá
with torch.no_grad():  # Không tính toán gradient
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
