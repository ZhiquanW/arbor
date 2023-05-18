import os
import sys


import numpy as np
import dearpygui.dearpygui as dpg

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import tags
import render


def registery_render_buffer(fig):
    w, h = fig.canvas.get_width_height()
    buffer = np.ones((h, w, 4), dtype=np.float32)
    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(
            width=w,
            height=h,
            default_value=buffer.flatten(),  # type: ignore
            format=dpg.mvFormat_Float_rgba,
            tag=tags.Texture.render,
        )
    return buffer, (w, h)
