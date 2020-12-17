import cv2
import os
import time

# setting the face picture save position
train_set = "data/trainSet"
test_set = "data/testSet"
# =======================================


def build_facedata(tag, window_name='please look the camera,press q to quit',
                      camera_idx=0):  # cite:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html "OpenCV-python Tutorials"
    cv2.namedWindow(window_name)
    # video input set,could input a local video,for now i use camera
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
        # Is the frame read correctly,Get a frame of data
        ret, frame = cap.read()
        if not ret:
            break
        # catch face in this frame
        opencv_getface(frame, tag)
        # setting quit hotkey
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def opencv_getface(frame,tag):  # cite: https://www.programcreek.com/python/example/79435/cv2.CascadeClassifier "example4"
    # setting face classier
    classfier = cv2.CascadeClassifier(
        "C:/Users/Yuyun/AppData/Local/Programs/Python/Python37/LIb/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # rectangular color
    color = (255, 0, 0)
    # Convert to grayscale image
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detect，scale and effective points
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0:  # >0 means detected human face
        # draw every human face in that picture
        for face_rects in face_rects:
            x, y, w, h = face_rects
            tmp = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # save image for training
            save_data(tmp, tag, num)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1


# check and create directory
def set_dir(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == 17:  # 17 means this file are already exists
            pass
        else:
            raise


# save in dir: ./trainSet/'tag'
def save_data(frame, tag, num):
    set_dir(os.path.join(train_set, str(tag)))
    filename = os.path.join(train_set, str(tag), f'{int(time.time())}_{num}.jpg')
    cv2.imwrite(filename, frame)
