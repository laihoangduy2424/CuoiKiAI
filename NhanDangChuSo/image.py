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

# Hàm dự đoán và hiển thị toàn bộ ảnh trong tập kiểm tra
def predict_all_images(model_path='model.pth'):
    # Tải dữ liệu kiểm tra
    _, testloader = load_data()

    # Khởi tạo mô hình và tải trọng số đã lưu
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Chuyển mô hình sang chế độ đánh giá

    # Duyệt qua tất cả các batch
    for i, (images, labels) in enumerate(testloader):
        # Dự đoán nhãn cho batch hiện tại
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # In ra nhãn thực tế và nhãn dự đoán
        print(f'\nBatch {i + 1}')
        print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(len(labels))))
        print('Predicted:   ', ' '.join('%5s' % predicted[j].item() for j in range(len(predicted))))

        # Hiển thị tất cả ảnh trong batch
        imshow(torchvision.utils.make_grid(images))

# Chạy hàm dự đoán để hiển thị toàn bộ 10,000 ảnh
if __name__ == '__main__':
    predict_all_images()
