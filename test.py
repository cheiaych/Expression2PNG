import expression_tracking
import cv2
from pygrabber.dshow_graph import FilterGraph

#filepath = input('Enter a filepath ("exit" to exit): ')
#Anger, Disgust, Fear, Happy, Sad, Suprise, Neutral
 
#expression_tracking.testAnalysis(filepath)

class device:
    def __init__(self, index, name):
        self.index = index
        self.name = name

def list_cams():
    devices = FilterGraph().get_input_devices()
    webcams = []
    for device_index, device_name in enumerate(devices):
        temp_device = device(device_index, device_name)
        webcams.append(temp_device)
    return webcams

list_cams = list_cams()
for d in list_cams:
    print("Index: %i Name: %s" % (d.index, d.name))

expression_tracking.custom_analysis(db_path = "TestImages",
    model_name = "VGG-Face",
    detector_backend = "opencv",
    distance_metric = "cosine",
    enable_face_analysis  = True,
    source = 0,
    time_threshold = 0.5,
    frame_threshold = 1)

"""
#
expression_tracking.analysis(db_path = "TestImages",
    model_name = "VGG-Face",
    detector_backend = "opencv",
    distance_metric = "cosine",
    enable_face_analysis  = True,
    source = 0,
    time_threshold = 0.5,
    frame_threshold = 1)

"""
