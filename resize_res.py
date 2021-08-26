import os
import cv2
from alive_progress import alive_bar


input_folder = "./imgs/validation"
with alive_bar(len(os.listdir(input_folder))) as bar:
    for i, file in enumerate(os.listdir(input_folder)):
        im = cv2.imread(os.path.join(input_folder, file))
        im = cv2.resize(im, (512,512))
        cv2.imwrite(os.path.join(input_folder, file), im)
        bar()