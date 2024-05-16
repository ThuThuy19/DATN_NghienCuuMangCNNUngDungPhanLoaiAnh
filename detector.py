from ultralytics import YOLO  # Import YOLO from ultralytics
import torch  # Import torch library
import pandas as pd  # Import pandas library with the alias pd

class YOLODetector:
    def __init__(self, path_model):
        # Xác định thiết bị để sử dụng (GPU hoặc CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.path_model = path_model
        self.model = self.load_model()  # Load mô hình khi khởi tạo

    def load_model(self):
        # Tải mô hình YOLO từ đường dẫn được cung cấp
        model = YOLO(self.path_model)
        return model

    def predict(self, frame):
        # Thực hiện dự đoán trên frame được cung cấp
        result = self.model(frame)[0]  # Lấy kết quả dự đoán
        boxes = result.boxes  # Truy cập vào thuộc tính boxes để lấy bounding boxes

        # Trích xuất thông tin cần thiết từ bounding boxes
        box_data = boxes.data.cpu().numpy()  # Chuyển đổi sang numpy array
        # Tạo DataFrame từ dữ liệu bounding boxes
        result_df = pd.DataFrame(box_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
        return result_df


