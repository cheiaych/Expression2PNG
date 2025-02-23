import expression_functions as tracking
import gui_functions
import cv2
import numpy
import math
from pygrabber.dshow_graph import FilterGraph

import dearpygui.dearpygui as dpg

def map_to_log_range(x):
    x_linear = (x + 100) / 200 * (math.log10(100) - math.log10(0.01)) + math.log10(0.01)
    x_mapped = 10 ** x_linear
    return x_mapped

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

average_count = 1
current_count = 0
demographies_list = []

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

#inertia method
'''while dpg.is_dearpygui_running():

    demographies = tracking.expression_analysis(
                    detector_backend = "opencv",
                    source = cam)

    for key in list(multipliers.keys()):
        multipliers[key] = map_to_log_range(dpg.get_value(key))
        demographies[key] = round(demographies[key] * multipliers[key], 2)

#Interia to help with rapid image changing
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

    dpg.render_dearpygui_frame()'''

#average method
while dpg.is_dearpygui_running():

    demographies = tracking.expression_analysis(
                    detector_backend = "opencv",
                    source = cam)

    for key in list(multipliers.keys()):
        multipliers[key] = map_to_log_range(dpg.get_value(key))
        demographies[key] = round(demographies[key] * multipliers[key], 2)

    average_count = dpg.get_value('inertia')

    if (current_count > average_count):
        avg_demographies = {
        'angry': 0.0, 
        'disgust': 0.0, 
        'fear': 0.0, 
        'happy': 0.0, 
        'sad': 0.0, 
        'surprise': 0.0, 
        'neutral': 0.0
        }
        for key in list(multipliers.keys()):
            demo_count = 0
            while demo_count < len(demographies_list):
                print(f"Demo count: {demo_count}, Current count: {current_count}, Demo_List length: {len(demographies_list)}")
                avg_demographies[key] = avg_demographies[key] + demographies_list[demo_count][key]
                demo_count = demo_count + 1
            avg_demographies[key] = round(avg_demographies[key] / average_count, 2)
        demographies_list.clear()

        expression = max(avg_demographies, key=avg_demographies.get)
        print ("Avg Demo")
        print(avg_demographies)

        if (expression != current_expression):
            gui_functions.display_expression(expression)

        current_expression = expression
        current_count = 0

    else:
        demographies_list.append(demographies)

    current_count = current_count + 1
    dpg.render_dearpygui_frame()

dpg.destroy_context()

