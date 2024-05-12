import expression_functions as tracking
import gui_functions
import cv2
import numpy
from pygrabber.dshow_graph import FilterGraph

import dearpygui.dearpygui as dpg

multipliers = {
    "angry": float(0.0),
    "disgust": float(0.0),
    "fear": float(0.0),
    "happy": float(0.0),
    "sad": float(0.0),
    "surprise": float(0.0),
    "neutral": float(0.0),
    }

inertia = 0.0

gui_functions.setup_dpg_window()

gui_functions.display_expression('neutral')

current_demographies = {
    'angry': 0.0, 
    'disgust': 0.0, 
    'fear': 0.0, 
    'happy': 0.0, 
    'sad': 0.0, 
    'surprise': 0.0, 
    'neutral': 0.0
    }

current_expression = 'neutral'

source = 0
cam = cv2.VideoCapture(source)
while dpg.is_dearpygui_running():

    demographies = tracking.expression_analysis(
                    detector_backend = "opencv",
                    source = cam)

    for key in list(multipliers.keys()):
        multipliers[key] = dpg.get_value(key)
        if (multipliers[key] > 0): 
            demographies[key] = round(demographies[key] * multipliers[key], 2)
        elif (multipliers[key] < 0):
            demographies[key] = round(demographies[key] / multipliers[key], 2)
        else:
            demographies[key] = round(demographies[key], 2) 

    inertia = dpg.get_value('inertia')

    change_expression = False
    for key in list(current_demographies.keys()):
        if (abs(demographies[key] - current_demographies[key]) > inertia):
            change_expression = True

    expression = max(demographies, key=demographies.get)
    print(demographies)

    if (change_expression and expression != current_expression):
        gui_functions.display_expression(expression)

    current_expression = expression

    dpg.render_dearpygui_frame()


dpg.destroy_context()

'''
dpg.create_context()

img = cv2.imread('./TestImages/TEST_PNG.png')
scaledimg = functions.scale_image(img, 1024, 500)
cv2.imshow('Output', scaledimg)
flatimg = functions.flatten_img(scaledimg)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(scaledimg.shape[1], scaledimg.shape[0], flatimg, tag='texture_tag', format=dpg.mvFormat_Float_rgb)

with dpg.window(tag='expression_display', autosize=True):
    dpg.add_image('expression_image')

with dpg.window(tag='Control Window', no_close=True, autosize=True):
        dpg.add_text('Sensitivities')
        dpg.add_slider_float(label='Angry', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Disgust', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Fear', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Happy', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Sad', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Suprise', default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label='Neutral', default_value=0.0, max_value=100.0, min_value=-100.0)

dpg.create_viewport(title='Window 1', width=1024, height=576)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('image_tag', True)

with dpg.item_handler_registry(tag='#resize_handler'):
    dpg.add_item_resize_handler(callback=functions.window_resized)

dpg.start_dearpygui()

dpg.destroy_context()
'''

"""
cam = cv2.VideoCapture(0)

_, img = cam.read()



while dpg.is_dearpygui_running():
    _, img = cam.read()

    data = numpy.flip(img, 2)
    data = data.ravel()
    data = numpy.asfarray(data, dtype='f')
    texture_data = numpy.true_divide(data, 255.0)
    dpg.set_value("texture_tag", texture_data)

    dpg.render_dearpygui_frame()

dpg.destroy_context()
"""

""" #
sensitivities = {
    "angry": 0.5,
    "disgust": 1.0,
    "fear": 1.0,
    "happy": 5.0,
    "sad": 0.5,
    "surprise": 1.0,
    "neutral": 1.5,
}


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
"""

""" #
expression_tracking.scan_for_cameras()

expression_tracking.begin_analysis(source = 0)
"""

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
