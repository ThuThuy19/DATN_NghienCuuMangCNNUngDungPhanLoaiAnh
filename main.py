from detector import yolo 
import numpy as np
import cv2



if __name__=='__main__':
    path_model='./train18/weights/best.pt'
    model=yolo(path_model)
    image=cv2.imread("./images/16.JPG")
    # assert image.shape
    clone_image=cv2.resize(image,(640,640))
    image=clone_image
    
    results= model.predict(image)
    contour_coordinates = []
    
    for index,row in results.iterrows():
        x,y,w,h=int(row[0]), int(row[1]), int(row[2])-int(row[0]), int(row[3])-int(row[1])
        roi = image[y:y+h, x:x+w]
        roi=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Result1", roi)
        ret, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over the contours and add the coordinates to the list
        for contour in contours:
            contour_coordinates.append(np.concatenate(contour) + (x, y))
    # print(contour_coordinates)        
            
            
    clone_image_= clone_image.copy()
    
   
    similarity_threshold = 0.03
    overlapping_contours = []
# Loop through each contour
    for i, contour in enumerate(contour_coordinates):
        # Loop through the remaining contours
        for j in range(i + 1, len(contour_coordinates)):
            # Calculate the shape similarity between the two contours
            if j < len(contour_coordinates):
                similarity = cv2.matchShapes(contour, contour_coordinates[j], cv2.CONTOURS_MATCH_I2, 0)

            # If the shape similarity is below the threshold, the contours are considered to overlap
            if similarity < similarity_threshold:
                # Draw a bounding box around the overlapping contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                x, y, w, h = cv2.boundingRect(contour_coordinates[j])
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # If the shape similarity is above the threshold, the contours are considered to be far apart
            elif similarity > (1.0 / 0.6):
                # Draw a bounding box around the far apart contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                x, y, w, h = cv2.boundingRect(contour_coordinates[j])
                cv2.rectangle(clone_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # else

    
    k=np.random.randint(0,2000)
    cv2.imwrite("/home/thinhdo/WorkSpace/NCKH/output/img_convert"+str(k)+".JPG",clone_image)
    cv2.imshow("Result", clone_image)
    for contour in contour_coordinates:
    # Draw the contour on the drawing image
        color = (0, 255, 1)
        cv2.drawContours(clone_image_, [contour], -1, color, 1)
    cv2.imwrite("/home/thinhdo/WorkSpace/NCKH/output/img_detect"+str(k)+".JPG",clone_image_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()