import os
import CNN
import cv2
import time
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw


title="who?|press q to quit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label = {
    0: "me",
    1: "roommate"
}


def sort_cnn(image):
    a= transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                               transforms.ToTensor(), transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                                                           std=[0.2, 0.2, 0.2])])
    pred_data = a(image)
    pred_data = pred_data.view(-1, 3, 32, 32)  # rebuild tensor
    cnn = CNN.CNN().to(device)
    cnn.load_state_dict(torch.load("data/model/trained.pkl"))  # Load model parameter weights
    sorted = cnn(pred_data.to(device))
    pred = sorted.max(1, keepdim=True)[1]
    return pred.item()


def paint_text_opencv(im, text, position, color):  # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/ geek for geeks
    PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("arial.ttf", 40, encoding="unic")
    fillColor = color
    draw = ImageDraw.Draw(PIL)
    draw.text(position, text, font=font, fill=fillColor)
    frame = cv2.cvtColor(np.asarray(PIL), cv2.COLOR_RGB2BGR)
    return frame


cv2.namedWindow(title, 0)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    classfier = cv2.CascadeClassifier(
        "C:/Users/Yuyun/AppData/Local/Programs/Python/Python37/LIb/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    color = (255, 0, 0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(face_rects) > 0:
        for face in face_rects:
            x, y, w, h = face
            tmp = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            PIL= Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            key = sort_cnn(PIL)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            frame = paint_text_opencv(frame, label[key], (x-10, y+h+10), color)
        cv2.imwrite("data/tmp/{}.jpg".format(int(time.time())), frame)


    cv2.resizeWindow(title, 1024, 768)
    cv2.imshow(title, frame)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

