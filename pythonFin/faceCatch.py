import cv2
import os
import time



# setting the face picture save position also it's training set
training_set = "data/trainingSet"
# setting testingset save position
testing_set = "data/testingSet"
# set the model save position
model_save = "data/model"


def building_facedata(tag, window_name='catch face', camera_idx=0):
    cv2.namedWindow(window_name)
    # video input set
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
       # Is the frame read correctly,Get a frame of data
        ret, frame = cap.read()
        if not ret:
            break
        # catch face in this frame
        opencv_getface(frame, tag)
        # setting quit hotkey
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
  # release camera amd windows
    cap.release()
    cv2.destroyAllWindows()


def opencv_getface(frame, tag):
    # setting face classier
    classfier = cv2.CascadeClassifier("C:/Users/Yuyun/AppData/Local/Programs/Python/Python37/LIb/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    # rectangular color
    color = (255, 0, 0)
    # Convert to grayscale image
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detect，scale and effective points
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0: # >0 means detected human face
        # draw every human face in that picture
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # save image for training
            save_face(image, tag, num)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1


EEXIST = 17


def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == EEXIST:
            pass
        else:
            raise


def save_face(image, tag, num):
  # create directory
    makedir_exist_ok(os.path.join(training_set, str(tag)))
    img_name = os.path.join(training_set, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
    # save in tag
    cv2.imwrite(img_name, image)