import torch
from model import SimpleCNN  # Thay thế bằng mô hình của bạn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình
model = SimpleCNN().to(device)

# Tải trọng số đã lưu
model.load_state_dict(torch.load('model1.pth', map_location=device))

# Chuyển mô hình sang chế độ đánh giá
model.eval()

# Chuẩn hóa ảnh (từ tập dữ liệu MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Hàm dự đoán từ tensor ảnh MNIST
def predict_from_tensor(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# Tải tập dữ liệu MNIST và chọn một ảnh từ tập test
from torchvision.datasets import MNIST
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

# Chọn một ảnh và label từ tập test
index = 9  # Bạn có thể thay đổi chỉ số để chọn ảnh khác
image, label = test_dataset[index]
image = image.unsqueeze(0)  # Thêm chiều batch

# Hiển thị ảnh
plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
plt.title(f"Ground Truth Label: {label}")
plt.show()

# Gọi hàm dự đoán với ảnh từ MNIST
prediction = predict_from_tensor(image)
print(f'Dự đoán của mô hình là: {prediction}')
