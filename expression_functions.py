import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions
from deepface.commons.logger import Logger
from pygrabber.dshow_graph import FilterGraph

logger = Logger(module="commons.realtime")

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks

class device:
    def __init__(self, index, name):
        self.index = index
        self.name = name

def scan_for_cameras():
    devices = FilterGraph().get_input_devices()
    webcams = []
    for device_index, device_name in enumerate(devices):
        temp_device = device(device_index, device_name)
        webcams.append(temp_device)
    return webcams

def expression_analysis(
    detector_backend="opencv",
    source=""):

    #cv2.namedWindow("Output", cv2.WINDOW_KEEPRATIO)
    _, cap_img = source.read()
    #cv2.imshow("Output", cap_img)
    """faces = DeepFace.extract_faces(
        img_path=cap_img,
        target_size=functions.find_target_size(model_name=model_name),
        detector_backend=detector_backend,
        enforce_detection=False
    )"""
    #face_dimensions = faces[0]["facial_area"]
    #print(face_dimensions)

    #cropped_face = cap_img[int(face_dimensions["y"]):int(face_dimensions["y"] + face_dimensions["h"]), int(face_dimensions["x"]):int(face_dimensions["x"] + face_dimensions["w"])]
    #cv2.imshow("Output", cap_img)
    
    demographies = DeepFace.analyze(
                            img_path=cap_img,
                            actions=("emotion"),
                            detector_backend=detector_backend,
                            enforce_detection=False,
                            silent=True,
                        )
    
    #print(demographies[0]["emotion"])
    return demographies[0]["emotion"]

def show_image(emotion=""):
        img = cv2.imread("./Testimages/" + emotion + ".jpg")
        img = cv2.resize(img, (500,500))
        cv2.imshow("Expression", img)

def begin_analysis(source):
    cam = cv2.VideoCapture(source)
    while True:
        emotion = expression_analysis(
        model_name = "VGG-Face",
        detector_backend = "opencv",
        source = cam)

        #print(emotion)
        show_image(emotion)

        if cv2.waitKey(500) & 0xFF == ord('q'):  #press q to quit, waitKey sets refresh rate in ms (waitKey(1000) is 1 frame a second), 
            break

    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()