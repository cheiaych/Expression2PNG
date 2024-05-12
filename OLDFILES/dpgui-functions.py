import cv2
import numpy
import dearpygui.dearpygui as dpg

def setup_main_gui():
    with dpg.window(tag="Primary Window"):

        dpg.add_image("texture_tag")
        
        dpg.add_text("Sensitivities")
        dpg.add_slider_float(label="Angry", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Disgust", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Fear", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Happy", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Sad", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Suprise", default_value=0.0, max_value=100.0, min_value=-100.0)
        dpg.add_slider_float(label="Neutral", default_value=0.0, max_value=100.0, min_value=-100.0)

        dpg.add_button(label="Start")

    dpg.create_viewport(title='Custom Title', width=800, height=400)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()

def close_gui():
    dpg.destroy_context()

def show_camera():
    frame_width = cv2.CAP_PROP_FRAME_WIDTH
    frame_height = cv2.CAP_PROP_FRAME_HEIGHT
    video_fps = cv2.CAP_PROP_FPS

    with dpg.window(tag="Primary Window"):
        dpg.add_image("texture_tag")

    data = numpy.flip(img, 2)
    data = data.ravel()
    data = numpy.asfarray(data, dtype='f')
    texture_data = numpy.true_divide(data, 255.0)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(img.shape[1], img.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)