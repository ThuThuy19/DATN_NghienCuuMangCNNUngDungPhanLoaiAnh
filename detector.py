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

    # def predict(self, frame):
    #     # Phương thức dự đoán các đối tượng trong frame được cung cấp
    #     result = self.model(frame)  # Dự đoán các đối tượng trong frame
    #     result = torch.Tensor.cpu(result[0].boxes.boxes)  # Chuyển từ tensor trên GPU sang CPU (nếu máy chỉ có GPU)
    #     result = pd.DataFrame(result).astype("float")  # Chuyển kết quả thành DataFrame của Pandas
    #     return result  # Trả về kết quả dự đoán


    # def predict(self, frame):
    #     # Phương thức dự đoán các đối tượng trong frame được cung cấp
    #     with torch.no_grad():  # Tắt việc tính toán đạo hàm cho quá trình dự đoán
    #         result = self.model(frame)  # Dự đoán các đối tượng trong frame
        
    #     # Trích xuất thông tin về các hộp từ kết quả dự đoán
    #     result_tensor = result[0].boxes.data
    #     # Chuyển tensor từ GPU sang CPU nếu cần thiết
    #     result_cpu = result_tensor.cpu() if self.device == 'cuda' else result_tensor
    #     # Chuyển kết quả thành DataFrame của Pandas
    #     result_df = pd.DataFrame(result_cpu.numpy(), columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])
        
    #     return result_df  # Trả về kết quả dự đoán dưới dạng DataFrame của Pandas

def predict(self, frame):
    # Phương thức dự đoán các đối tượng trong frame được cung cấp
    with torch.no_grad():  # Tắt việc tính toán đạo hàm cho quá trình dự đoán
        results = self.model(frame)  # Dự đoán các đối tượng trong frame

    all_boxes = []  # Danh sách chứa thông tin về các hộp từ tất cả các kết quả dự đoán

    # Lặp qua từng kết quả dự đoán trong danh sách
    for result in results:
        boxes_tensor = result.xyxy[0]  # Lấy tensor chứa thông tin về các hộp từ kết quả dự đoán
        # Chuyển từ tensor trên GPU sang CPU nếu cần thiết
        boxes_numpy = boxes_tensor.cpu().numpy() if self.device == 'cuda' else boxes_tensor.numpy()
        all_boxes.append(boxes_numpy)  # Thêm thông tin về các hộp vào danh sách

    # Tạo DataFrame từ danh sách chứa thông tin về các hộp từ tất cả các kết quả dự đoán
    result_df = pd.DataFrame(np.concatenate(all_boxes), columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class"])
    return result_df  # Trả