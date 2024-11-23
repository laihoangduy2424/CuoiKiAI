import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN  # Sử dụng mô hình `SimpleCNN`
from data_preparation import load_data  # Sử dụng hàm `load_data` để tải dữ liệu

# Hàm hiển thị ảnh (sau khi unnormalize)
def imshow(img):
    img = img / 2 + 0.5  # Bỏ chuẩn hóa để hiển thị ảnh rõ ràng
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Hàm dự đoán và hiển thị nhiều batch
def predict_batches(model_path='model1.pth', num_batches=3):
    # Tải dữ liệu kiểm tra
    _, test_loader = load_data()  # Giả sử hàm trả về 2 giá trị

    # Khởi tạo mô hình và tải trọng số đã lưu
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá

    # Duyệt qua các batch
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_batches:  # Giới hạn số batch muốn hiển thị
            break

        # Dự đoán nhãn cho batch hiện tại
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # In ra nhãn thực tế và nhãn dự đoán
        print(f'\nBatch {i+1}')
        print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(min(len(labels), 8))))
        print('Predicted:   ', ' '.join('%5s' % predicted[j].item() for j in range(min(len(predicted), 8))))

        # Hiển thị 8 ảnh đầu tiên trong batch
        imshow(torchvision.utils.make_grid(images[:8]))

# Chạy hàm dự đoán cho nhiều batch
if __name__ == '__main__':
    predict_batches(num_batches=10)  # Hiển thị 10 batch đầu tiên
