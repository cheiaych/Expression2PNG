import expression_functions
import cv2
import numpy
from pygrabber.dshow_graph import FilterGraph

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo

class image:
    def __init__(self, data, width, height):
        self.data = data
        self.width = width
        self.height = height

expressions_list = {'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'}
expressions = {'angry': image, 'disgust': image, 'fear': image, 'happy': image, 'neutral': image, 'sad': image, 'surprise': image}
expression_texture_id = None
current_expression = 'neutral'


def scale_texture(tag, max_width, max_height):
    ratio = min(max_width / dpg.get_item_width(tag), max_height / dpg.get_item_height(tag))
    dpg.add_image(tag, width = int(dpg.get_item_width(tag) * ratio), height = int(dpg.get_item_height(tag) * ratio))

def window_resized():
     print('Resized')
     """
     img = cv2.imread('./TestImages/TEST_PNG.png')
     scaledimg = scale_image(img, dpg.get_viewport_client_width(), dpg.get_viewport_client_height())
     flatimg = flatten_img(scaledimg)
     with dpg.texture_registry(show=False):
        dpg.add_raw_texture(scaledimg.shape[1], scaledimg.shape[0], flatimg, tag = "texture_tag", format=  dpg.mvFormat_Float_rgb)
    """

def load_expressions():
    for expression in expressions:
        print('./TestImages/' + expression + '.jpg')
        width, height, channels, data = dpg.load_image('./TestImages/' + expression + '.jpg')
        expressions[expression] = image(data, width, height)
                                  
def display_expression(expression):
    viewport_width = dpg.get_viewport_client_width() - 25
    viewport_height = dpg.get_viewport_client_height() - 25    

    ratio = min(viewport_width / expressions[expression].width, viewport_height / expressions[expression].height) 

    print('window height: ' + str(viewport_height) + ' window width: ' + str(viewport_width))
    #print('image height: ' + str(expressions[expression].height) + ' image width: ' + str(expressions[expression].width))
    #print('ratio: ' + str(ratio))


    try:
        dpg.delete_item('expression_display', children_only = True)
        dpg.delete_item(expression_texture_id)
    except:
        pass

    with dpg.texture_registry(show = False):
        expression_texture_id = dpg.add_dynamic_texture(width = expressions[expression].width, height = expressions[expression].height, default_value = expressions[expression].data)
        dpg.add_image(expression_texture_id, width = int(expressions[expression].width * ratio), height = int(expressions[expression].height * ratio), tag = 'expression', parent = 'expression_display')

def reset():
    for expression in expressions_list:
        dpg.set_value(expression, 0.0)
    dpg.set_value('inertia', 0.0)

def test_display():
    display_expression('neutral')

def setup_dpg_window():
    load_expressions()

    dpg.create_context()

    with dpg.window(tag = 'expression_display', autosize = True) as expression_display:
        dpg.add_text('Placeholder')

    with dpg.window(tag = 'control_display', no_close = True, autosize = True):
            
            #dpg.add_button(label="Test", callback=test_display)

            dpg.add_text('Multipliers')
            dpg.add_slider_float(tag = 'angry', label = 'Angry', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'disgust', label = 'Disgust', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'fear', label = 'Fear', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'happy', label = 'Happy', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'sad', label = 'Sad', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'surprise', label = 'Suprise', default_value = 0.0, max_value = 100.0, min_value = -100.0)
            dpg.add_slider_float(tag = 'neutral', label = 'Neutral', default_value = 0.0, max_value = 100.0, min_value = -100.0)

            dpg.add_text('Inertia')
            dpg.add_slider_int(tag = 'inertia', label = 'Intertia', default_value = 1, max_value = 100, min_value = 1)

            dpg.add_button(label = "Reset", callback = reset)


    with dpg.item_handler_registry(tag = '#resize_handler'):
        dpg.add_item_resize_handler(callback = window_resized())

    dpg.set_primary_window('expression_display', True)
    
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (26, 26, 60), category=dpg.mvThemeCat_Core)

    dpg.bind_item_theme(expression_display, theme)

    dpg.create_viewport(title = 'Window 1', width = 1024, height = 576)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    #display_expression('happy')

