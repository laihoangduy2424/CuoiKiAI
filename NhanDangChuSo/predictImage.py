import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # Import matplotlib để hiển thị ảnh
from model import SimpleCNN  # Đảm bảo mô hình đã được định nghĩa trong model.py

# Thiết lập thiết bị (GPU nếu có, nếu không sử dụng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hàm chuẩn bị ảnh đầu vào
def prepare_image(image_path):
    # Đọc ảnh và chuyển về grayscale nếu cần
    img = Image.open(image_path).convert('L')  # 'L' để chuyển thành ảnh xám (grayscale)

    # Các phép biến đổi (resize, chuyển thành tensor, chuẩn hóa)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize về kích thước 28x28 (kích thước chuẩn của MNIST)
        transforms.ToTensor(),  # Chuyển ảnh thành tensor và chuẩn hóa giá trị pixel về [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Chuẩn hóa dữ liệu về khoảng [-1, 1] (theo cách huấn luyện mô hình)
    ])

    img_tensor = transform(img)  # Chuyển đổi ảnh thành tensor
    img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension (một ảnh duy nhất, nên kích thước tensor sẽ là (1, 1, 28, 28))

    # Hiển thị ảnh sau khi biến đổi
    img_for_display = img_tensor.squeeze(0)  # Loại bỏ batch dimension
    img_for_display = img_for_display * 0.5 + 0.5  # Đưa giá trị ảnh về khoảng [0, 1] để hiển thị
    plt.imshow(img_for_display.squeeze(0), cmap='gray')
    plt.title("Transformed Image")
    plt.axis("off")
    plt.show()

    return img_tensor

# Hàm dự đoán chữ số trên ảnh
def predict(image_path):
    # Tải mô hình đã huấn luyện
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('model1.pth', map_location=device, weights_only= True))  # Tải trọng số đã lưu

    # Chuẩn bị ảnh
    img_tensor = prepare_image(image_path)

    # Dự đoán
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    with torch.no_grad():  # Không tính toán gradient
        img_tensor = img_tensor.to(device)  # Đưa tensor ảnh vào device (GPU/CPU)
        output = model(img_tensor)  # Tiến hành dự đoán
        _, predicted = torch.max(output, 1)  # Chọn giá trị dự đoán có xác suất cao nhất

    return predicted.item()  # Trả về giá trị dự đoán

# Chạy dự đoán trên một ảnh
if __name__ == '__main__':
    image_path = 'anh11.jpg'  # Đường dẫn tới ảnh cần dự đoán
    predicted_digit = predict(image_path)
    print(f'Dự đoán chữ số: {predicted_digit}')
