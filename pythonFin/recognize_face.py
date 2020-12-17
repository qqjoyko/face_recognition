import os
import cv2
import time
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from CNN import CNN
from data_training import image_process, device, model_save, DEFAULT_MODEL


# set the face name, key correspond to the node with the largest output probability of the fully connected layer
label = {
    0: "me",
    1: "roommate"
}


def predict_model(image):
    a= image_process()
    pred_data = a(image)
    pred_data = pred_data.view(-1, 3, 32, 32)  # rebuild tensor
    net = CNN().to(device)
    net.load_state_dict(torch.load(os.path.join(model_save, DEFAULT_MODEL)))  # Load model parameter weights
    result = net(pred_data.to(device))
    pred = result.max(1, keepdim=True)[1]
    return pred.item()


def recognize_face(window_name='face recognize|press q to quit', camera_idx=0):
    cv2.namedWindow(window_name, 0)

    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        catch_frame = opencv_getface2(frame)

        cv2.resizeWindow(window_name, 1024, 768)
        cv2.imshow(window_name, catch_frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def opencv_getface2(frame):
    classfier = cv2.CascadeClassifier(
        "C:/Users/Yuyun/AppData/Local/Programs/Python/Python37/LIb/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    color = (255, 0, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:
        for face in face_rects:
            x, y, w, h = face
            tmp = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            PIL= toPIL(tmp)
            key = predict_model(PIL)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            frame = paint_text_opencv(frame, label[key], (x-10, y+h+10), color)
        cv2.imwrite("data/tmp/{}.jpg".format(int(time.time())), frame)
    return frame


# transfer frame to PIL
def toPIL(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def paint_text_opencv(im, text, position, color):  # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/ geek for geeks
    PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("arial.ttf", 40, encoding="unic")
    fillColor = color
    draw = ImageDraw.Draw(PIL)
    draw.text(position, text, font=font, fill=fillColor)
    frame = cv2.cvtColor(np.asarray(PIL), cv2.COLOR_RGB2BGR)
    return frame