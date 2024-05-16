from detector import yolo  
import numpy as np  
import cv2  
from sklearn.cluster import KMeans  

if __name__ == '__main__':
    path_model = './train18/weights/best.pt'  # Đường dẫn tới tập tin trọng số của mô hình YOLO
    model = yolo(path_model)  # Tạo một đối tượng yolo với mô hình YOLO được tải từ đường dẫn
    image = cv2.imread("./images/5.JPG")  # Đọc ảnh từ đường dẫn
    clone_image = cv2.resize(image, (640, 640))  # Resize ảnh về kích thước 640x640
    image = clone_image  # Gán ảnh đã resize cho biến image
    
    results = model.predict(image)  # Dự đoán các đối tượng trong ảnh
    contour_coordinates = []
    # Khởi tạo một mảng chứa tọa độ trung tâm của các đối tượng
    points = np.zeros((len(results.iterrows()), 2))
    
    # Lặp qua các kết quả dự đoán
    for index, row in results.iterrows():
        x, y, w, h = int(row[0]), int(row[1]), int(row[2]) - int(row[0]), int(row[3]) - int(row[1])
        # Tính toán tọa độ trung tâm của đối tượng và lưu vào mảng points
        points[index] = [(x + w/2), (y + h/2)]
    
    # Sử dụng thuật toán KMeans để phân cụm các điểm trung tâm thành 2 nhóm
    k_means = KMeans(n_clusters=2, random_state=0).fit(points)
    
    # Nhãn của các nhóm
    labels = k_means.labels_
    
    # Tạo các danh sách để lưu các đối tượng thuộc từng nhóm
    group_1 = []
    group_2 = []
    
    # Lặp qua các kết quả dự đoán để phân loại các đối tượng vào nhóm tương ứng
    for index, row in results.iterrows():
        x, y, w, h = int(row[0]), int(row[1]), int(row[2]) - int(row[0]), int(row[3]) - int(row[1])
        # Kiểm tra nhãn của nhóm và thêm đối tượng vào danh sách tương ứng
        if labels[index] == 0:
            group_1.append((x+w/2, y+h/2, w, h))
        else:
            group_2.append((x+w/2, y+h/2, w, h))
    
    # Tính trung bình các tọa độ của các đối tượng trong từng nhóm
    group_1_mean = np.mean(group_1, axis=0)
    group_2_mean = np.mean(group_2, axis=0)
    

