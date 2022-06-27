import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN

import warnings
warnings.filterwarnings("ignore")


# Sex = {'1' : 'Female', '0': 'Male'}
# Race = {'0': 'White', '1': 'Black', '2': 'Asian', '3':'Indian', '4' : 'Others'}


def process_img(name, img_size):
    img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    face = MTCNN().detect_faces(img)       # Assuming only one face is given
    if len(face) != 0:
        x1, y1, width, height = face[0]['box']
        x2, y2 = x1 + width, y1 + height
        new_img = img[y1:y2, x1:x2]
    else:
        new_img = img

    new_img = cv2.resize(new_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    new_img = np.array(new_img) / 255.0
    return  new_img
