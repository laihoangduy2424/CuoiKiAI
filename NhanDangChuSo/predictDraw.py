import torch
from PIL import Image, ImageGrab
import torchvision.transforms as transforms
import numpy as np
import tkinter as tk
from model import SimpleCNN  # Đảm bảo mô hình đã được định nghĩa trong model.py

# Thiết lập thiết bị (GPU nếu có, nếu không sử dụng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình và tải trọng số đã huấn luyện
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('model1.pth', map_location=device))  # Tải trọng số đã lưu
model.eval()

# Hàm chuẩn bị ảnh cho dự đoán
def prepare_image(img):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = img.convert('L')  # Đổi thành ảnh xám
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(device)

# Hàm dự đoán chữ số
def predict(img):
    img_tensor = prepare_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Lưu ảnh từ canvas và dự đoán
def save_and_predict():
    # Lấy tọa độ canvas và chuyển thành ảnh
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Lưu ảnh canvas
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    digit = predict(img)
    result_label.config(text=f'Dự đoán: {digit}')

# Vẽ trên canvas
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x + 8, y + 8, fill='black', width=5)

# Xóa canvas
def clear_canvas():
    canvas.delete('all')
    result_label.config(text="Dự đoán: ")

# Thiết lập giao diện Tkinter
root = tk.Tk()
root.title("Vẽ chữ số và dự đoán")
root.geometry("300x400")

canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.pack(pady=20)
canvas.bind("<B1-Motion>", draw)

# Nút dự đoán
predict_button = tk.Button(root, text="Dự đoán", command=save_and_predict)
predict_button.pack()

# Nút xóa canvas
clear_button = tk.Button(root, text="Xóa", command=clear_canvas)
clear_button.pack()

# Hiển thị kết quả dự đoán
result_label = tk.Label(root, text="Dự đoán: ", font=("Helvetica", 16))
result_label.pack(pady=20)

root.mainloop()
