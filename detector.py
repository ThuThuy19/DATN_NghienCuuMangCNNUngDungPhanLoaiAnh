
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from time import time
import supervision as sv
import pandas as pd
class yolo:
    def __init__(self, path_model) -> None:
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.path_model= path_model
        self.model=self.load_model()
         
    def load_model(self):
        model = YOLO(self.path_model)  # load a pretrained YOLOv8n model
        model.fuse()
        return model
    def predict(self,frame):
        result = self.model(frame)
        result=torch.Tensor.cpu(result[0].boxes.boxes)   # For machine has GPU only such as Jeson Nano
        result=pd.DataFrame(result).astype("float")
       
        return result
    