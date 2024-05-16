from detector import YOLODetector 
import numpy as np
import cv2 


if __name__ == '__main__':
    # Load mô hình YOLO
    path_model = './train18/weights/best.pt'
    model = YOLODetector(path_model)
     # Đọc ảnh đầu vào
    image = cv2.imread("./images/5.JPG")

    # Resize ảnh về kích thước 640x640
    clone_image = cv2.resize(image, (640, 640))
    image = clone_image

    # Dự đoán các đối tượng trong ảnh sử dụng mô hình YOLO
    results = model.predict(image)

    # Khởi tạo danh sách để lưu trữ tọa độ contour
    contour_coordinates = []
      # Duyệt qua các đối tượng đã dự đoán
    for index, row in results.iterrows():
        # Trích xuất tọa độ bounding box của đối tượng
        x, y, w, h = int(row[0]), int(row[1]), int(row[2]) - int(row[0]), int(row[3]) - int(row[1])

        # Trích xuất vùng quan tâm (ROI)
        roi = image[y:y + h, x:x + w]

        # Chuyển đổi ROI sang ảnh grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Áp dụng ngưỡng để thu được ảnh nhị phân
        ret, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)

        # Tìm các contour trong ảnh nhị phân
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           # Duyệt qua các contour và thêm tọa độ vào danh sách
        for contour in contours:
            contour_coordinates.append(np.concatenate(contour) + (x, y))
             # Sao chép ảnh để vẽ bounding box
    clone_image_ = clone_image.copy()

    # Thiết lập ngưỡng tương đồng cho các contour chồng lấn
    similarity_threshold = 0.03
    overlapping_contours = []
      # Duyệt qua tọa độ contour
    for i, contour in enumerate(contour_coordinates):
        # Loop through the remaining contours
        for j in range(i + 1, len(contour_coordinates)):
             # Tính độ tương đồng hình dạng giữa các contour
            if j < len(contour_coordinates):
                similarity = cv2.matchShapes(contour, contour_coordinates[j], cv2.CONTOURS_MATCH_I2, 0)
                 # Nếu độ tương đồng nhỏ hơn ngưỡng, vẽ bounding box cho các contour chồng lấn
            if similarity < similarity_threshold:
                # Draw a bounding box around the overlapping contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                x, y, w, h = cv2.boundingRect(contour_coordinates[j])
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                 # Nếu độ tương đồng lớn hơn ngưỡng, vẽ bounding box cho các contour xa nhau
            elif similarity > (1.0 / 0.6):
                # Draw a bounding box around the far apart contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                x, y, w, h = cv2.boundingRect(contour_coordinates[j])
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                  # Lưu ảnh với bounding box
    k = np.random.randint(0, 2000)
    # cv2.imwrite("/home/thinhdo/WorkSpace/NCKH/output/img_convert" + str(k) + ".JPG", clone_image)
    cv2.imwrite("./output/img_convert" + str(k) + ".JPG", clone_image)
      # Hiển thị ảnh với bounding box
    cv2.imshow("Result", clone_image)

    # Vẽ contour trên ảnh gốc
    for contour in contour_coordinates:
        color = (0, 255, 1)
        cv2.drawContours(clone_image_, [contour], -1, color, 1)
           # Lưu ảnh với contour
    # cv2.imwrite("/home/thinhdo/WorkSpace/NCKH/output/img_detect" + str(k) + ".JPG", clone_image_)
    cv2.imwrite("./output/img_detect" + str(k) + ".JPG", clone_image_)

    # Đợi người dùng nhập để đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()