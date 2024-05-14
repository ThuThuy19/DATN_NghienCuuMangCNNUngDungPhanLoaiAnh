from ultralytics import YOLO  
import torch  
import pandas as pd  

class YOLODetector:
    def __init__(self, path_model):
        # Determine the device to use (GPU or CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.path_model = path_model
        self.model = self.load_model()

    def load_model(self):
        # Load the YOLO model from the provided path
        model = YOLO(self.path_model)
        return model

    def predict(self, frame):
        # Perform prediction on the provided frame
        result = self.model(frame)[0]
        boxes = result.boxes  # Accessing the boxes attribute for bounding boxes

        # Extract the necessary information from the boxes
        box_data = boxes.data.cpu().numpy()  # Move to CPU and convert to numpy array
        # Ensure the correct columns are used based on the model's output format
        result_df = pd.DataFrame(box_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
        return result_df


