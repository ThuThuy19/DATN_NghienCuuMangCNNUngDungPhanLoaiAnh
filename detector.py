from ultralytics import YOLO
import torch
import pandas as pd

class YOLODetector:
    def __init__(self, path_model) -> None:
        # Kiểm tra xem GPU có khả dụng không, nếu không sử dụng CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # In ra thông điệp xác định thiết bị được sử dụng
        print("Using Device: ", self.device)
        # Lưu đường dẫn đến mô hình YOLO được cung cấp
        self.path_model = path_model
        # Tải mô hình YOLO và kết hợp các phần của nó
        self.model = self.load_model()

    def load_model(self):
        # Tải mô hình YOLO từ đường dẫn được cung cấp
        model = YOLO(self.path_model)
        # Kết hợp các phần của mô hình để tăng tốc độ dự đoán
        model.fuse()
        # Trả về mô hình đã tải
        return model

    def predict(self, frame):
        # Thực hiện dự đoán trên frame được cung cấp
        result = self.model(frame)[0]
        # Trích xuất thông tin bounding boxes từ kết quả dự đoán và chuyển thành numpy array
        boxes_data = result.boxes.data.cpu().numpy()
        # Tạo DataFrame từ dữ liệu bounding boxes và đặt tên cho các cột
        result_df = pd.DataFrame(boxes_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"]).astype("float")
        # Trả về DataFrame chứa các bounding boxes dự đoán
        return result_df
