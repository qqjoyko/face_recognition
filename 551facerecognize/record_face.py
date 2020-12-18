import cv2
import os
import time


yourname = "cyy"
title = "look here|press q quit"

def splitandsave(frame, folder, faces, distribute):
    if distribute % 100 == 0:
        try:
            os.makedirs(os.path.join("data/train_set", str(folder)))
        except OSError as e:
            if e.errno == 17:  # 17 means this file are already exists
                pass
            else:
                raise
        imagename = os.path.join('data/test_set', str(folder), f'{int(time.time())}_{faces}.jpg')
    else:
        try:
            os.makedirs(os.path.join("data/test_set", str(folder)))
        except OSError as e:
            if e.errno == 17:  # 17 means this file are already exists
                pass
            else:
                raise
        imagename = os.path.join('data/train_set', str(folder), f'{int(time.time())}_{faces}.jpg')
    cv2.imwrite(imagename, frame)




cv2.namedWindow(title)
# video input set,could input a local video,for now i use camera
cap = cv2.VideoCapture(0)
distribute=0 #distribute test amd training data
while cap.isOpened():#get face from frame
    # Is the frame read correctly,Get a frame of data
    ret, frame = cap.read()
    if not ret:
        break
    # catch face in this frame
    haarcascade_frontalface = cv2.CascadeClassifier( # cite: https://www.programcreek.com/python/example/79435/cv2.CascadeClassifier "example4"
        "C:/Users/Yuyun/AppData/Local/Programs/Python/Python37/LIb/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # rectangular color
    # Convert to grayscale image
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detect，scale and effective points
    face_rects = haarcascade_frontalface.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    No_faces = 1
    if len(face_rects) > 0:  # >0 means detected human face
        rect_color = (255, 0, 0)
        # draw every human face in that picture
        for face_rects in face_rects:
            x, y, w, h = face_rects
            tmp = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # save image for training
            splitandsave(tmp, yourname, No_faces, distribute)
            distribute += 1
            print("catch",distribute/20,"(recommend at least 20.00+)")
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), rect_color, 2)
            No_faces += 1
    # setting quit hotkey
    cv2.resizeWindow(title, 800, 600)
    cv2.imshow(title, frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

