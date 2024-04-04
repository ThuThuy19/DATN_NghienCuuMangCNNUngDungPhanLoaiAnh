from detector import yolo 
import numpy as np
import cv2
from sklearn.cluster import KMeans


if __name__=='__main__':
    path_model='./train18/weights/best.pt'
    model=yolo(path_model)
    image=cv2.imread("/home/thinhdo/WorkSpace/NCKH/images/136.JPG")
    # assert image.shape
    clone_image=cv2.resize(image,(640,640))
    image=clone_image
    
    results= model.predict(image)
    contour_coordinates = []
    points = np.zeros((len(results.iterrows()), 2))
    for index,row in results.iterrows():
        x,y,w,h=int(row[0]), int(row[1]), int(row[2])-int(row[0]), int(row[3])-int(row[1])
        points[index] = [(x + w/2), (y + h/2)]
    
    k_means = KMeans(n_clusters=2, random_state=0).fit(points)
          
    labels = k_means.labels_
    group_1 = []
    group_2 = []
    for index,row in results.iterrows():
        x,y,w,h=int(row[0]), int(row[1]), int(row[2])-int(row[0]), int(row[3])-int(row[1])
        if labels[i] == 0:
            group_1.append((x+w/2, y+h/2, w, h))
        else:
            group_2.append((x+w/2, y+h/2, w, h))
    group_1_mean = np.mean(group_1, axis=0)     
    group_2_mean = np.mean(group_2, axis=0)