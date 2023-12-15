import cvzone
from ultralytics import YOLO
import cv2




video = r'C:\Users\Admin\Desktop\testvids/2.mp4'

cap = cv2.VideoCapture(video)
facemodel = YOLO('yolov8m-drone.pt')


while cap.isOpened():
    rt, video = cap.read()
    video = cv2.resize(video, (1020, 720))
    mainvideo = video.copy()

    face_result = facemodel.predict(video)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2-y1,x2-x1

            cvzone.cornerRect(video,[x1,y1,w,h],l=9,rt=3)
            cvzone.cornerRect(mainvideo, [x1, y1, w, h], l=9, rt=3)

            face = video[y1:y1+h,x1:x1+w]
            face = cv2.blur(face,(30,30))
            video[y1:y1+h,x1:x1+w] = face


    allFeeds = cvzone.stackImages([mainvideo,video],2,0.70)
    cv2.imshow('frame',mainvideo)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()