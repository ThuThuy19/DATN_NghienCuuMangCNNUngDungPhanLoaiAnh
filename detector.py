from ultralytics import YOLO  
import torch  
import numpy as np  
import cv2  
from time import time  
import supervision as sv  
import pandas as pd  

class yolo:
    def __init__(self, path_model) -> None:
        # Phương thức khởi tạo, được gọi khi một đối tượng yolo được tạo ra
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Xác định thiết bị sử dụng (GPU hoặc CPU)
        print("Using Device: ", self.device)  # In thông báo về thiết bị sử dụng
        self.path_model = path_model  # Lưu đường dẫn đến tập tin trọng số của mô hình
        self.model = self.load_model()  # Tải mô hình

    def load_model(self):
        # Phương thức để tải mô hình YOLO từ đường dẫn được cung cấp
        model = YOLO(self.path_model)  # Tải mô hình YOLO
        model.fuse()  # Kết hợp các lớp của mô hình
        return model  # Trả về mô hình đã tải

    def predict(self, frame):
        # Phương thức dự đoán các đối tượng trong frame được cung cấp
        result = self.model(frame)  # Dự đoán các đối tượng trong frame
        result = torch.Tensor.cpu(result[0].boxes.boxes)  # Chuyển từ tensor trên GPU sang CPU (nếu máy chỉ có GPU)
        result = pd.DataFrame(result).astype("float")  # Chuyển kết quả thành DataFrame của Pandas
        return result  # Trả về kết quả dự đoán
     




    