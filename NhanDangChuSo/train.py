# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from data_preparation import load_data
from model import SimpleCNN

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải dữ liệu
train_loader, val_loader, test_loader = load_data()  # Cập nhật để nhận cả val_loader

# Khởi tạo mô hình
model = SimpleCNN().to(device)

# Định nghĩa tiêu chuẩn và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hàm đánh giá trên tập validation
def evaluate(model, val_loader):
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    val_loss = 0
    correct = 0
    with torch.no_grad():  # Tắt tính toán gradient
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)  # Tổng loss
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)  # Tính loss trung bình
    accuracy = 100. * correct / len(val_loader.dataset)  # Tính độ chính xác
    return val_loss, accuracy

# Huấn luyện mô hình
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Chuyển mô hình sang chế độ huấn luyện
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Tiến hành forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Tiến hành backward pass và tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Đánh giá mô hình trên tập validation sau mỗi epoch
    val_loss, val_accuracy = evaluate(model, val_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {loss.item():.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Lưu trọng số của mô hình sau khi huấn luyện
torch.save(model.state_dict(), 'model1.pth')
print("Model weights saved to 'model.pth'")
