import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import torch
import array
import numpy as np

import matplotlib.pyplot as plt
import tags
import config
import callbacks
import context

import rlvortex.envs.base_env as base_env
import rlvortex

import tree_envs
import sim.torch_arbor as arbor


class Dearbor:
    def __init__(self) -> None:
        self.gui_awake()
        self.logical_awake()
        self.gui_init()

    def start(self):
        self.gui_start()

    def gui_awake(self):
        dpg.create_context()
        dpg.create_viewport(
            title="Dearbor",
            width=config.viewport_size[0],
            height=config.viewport_size[1],
            min_width=800,
            min_height=800,
        )
        dpg.setup_dearpygui()
        default_theme = config.create_default_theme_imgui()
        dpg.bind_theme(default_theme)

    def logical_awake(self):
        self.fig_size = (24, 8)
        self.fig_dpi = 256
        self.f = plt.figure(figsize=self.fig_size, dpi=self.fig_dpi)
        self.tree_ax = self.f.add_subplot(121, projection="3d")  # type: ignore
        self.energy_ax = self.f.add_subplot(122, projection="3d")  # type: ignore
        self.colorbar_p = None

        self.render_buffer, self.buffer_wh = context.registery_render_buffer(self.f)
        arbor_engine = arbor.TorchArborEngine(
            max_steps=20,
            max_branches_num=50,
            move_dis_range=[0.3, 0.5],
            move_rot_range=[[-10, 10], [-30, 30], [-10, 10]],
            new_branch_rot_range=[20, 40],
            node_branch_prob_range=[0.1, 0.5],
            node_sleep_prob_range=[0.001, 0.01],
            device=torch.device("cpu"),
        )
        self.env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
            env=tree_envs.CoreTorchEnv(arbor_engine=arbor_engine)
        )
        self.nodes_energy = None
        self.env_wrapper.awake()
        self.env_wrapper.reset()

    def gui_init(self):
        with dpg.window(tag=tags.Window.primary):
            self.__create_arbor_editor_window()
            self.__create_render_window()

    def gui_start(self):
        dpg.show_viewport()
        dpg.set_primary_window(tags.Window.primary, True)
        dpg.start_dearpygui()

    def __create_arbor_editor_window(self):
        with dpg.window(
            tag=tags.Window.operator,
            width=config.arbor_editor_width,
            height=1200,
            no_title_bar=True,
            no_close=True,
            no_collapse=True,
            no_move=True,
            no_resize=True,
        ):
            with dpg.collapsing_header(
                tag=tags.CollapsingHeader.arbor_editor,
                label="ArborEditor",
                default_open=True,
            ):
                with dpg.group():
                    dpg.add_text("max steps")
                    dpg.add_drag_int(
                        tag=tags.Drag.max_steps,
                        default_value=20,
                        min_value=1,
                        max_value=100,
                        width=config.drag_width,
                    )
                with dpg.group():
                    dpg.add_text("max branches num")
                    dpg.add_drag_int(
                        tag=tags.Drag.max_branches_num,
                        default_value=50,
                        min_value=1,
                        max_value=100,
                        width=config.drag_width,
                    )
                with dpg.group():
                    dpg.add_text("move distance range")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.move_distance_range,
                        default_value=[0.3, 0.5],
                        min_value=0,
                        max_value=1,
                        size=2,
                        width=config.drag_width,
                    )
                with dpg.group():
                    dpg.add_text("move direction rotation range (x axis)")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.move_rot_range_x,
                        default_value=[-10, 10],
                        min_value=-180,
                        max_value=180,
                        size=2,
                        width=config.drag_width,
                    )

                with dpg.group():
                    dpg.add_text("move direction rotation range (y axis)")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.move_rot_range_y,
                        default_value=[-40, 40],
                        min_value=-180,
                        max_value=180,
                        size=2,
                        width=config.drag_width,
                    )

                with dpg.group():
                    dpg.add_text("move direction rotation range (z axis)")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.move_rot_range_z,
                        default_value=[-10, 10],
                        min_value=-180,
                        max_value=180,
                        size=2,
                        width=config.drag_width,
                    )

                with dpg.group():
                    dpg.add_text("new branch rotation range")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.new_branch_rot_range,
                        default_value=[20, 40],
                        min_value=-180,
                        max_value=180,
                        size=2,
                        width=config.drag_width,
                    )
                with dpg.group():
                    dpg.add_text("node branch prob range")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.node_branch_prob_range,
                        default_value=[0.1, 0.5],
                        min_value=0,
                        max_value=1,
                        size=2,
                        width=config.drag_width,
                    )
                with dpg.group():
                    dpg.add_text("node sleep prob range")
                    dpg.add_drag_floatx(
                        tag=tags.Drag.node_sleep_prob_range,
                        default_value=[0.001, 0.01],
                        min_value=0,
                        max_value=1,
                        size=2,
                        width=config.drag_width,
                    )

            with dpg.collapsing_header(
                tag=tags.CollapsingHeader.energy,
                label="Energy System",
                default_open=True,
            ):
                with dpg.group():
                    with dpg.group():
                        dpg.add_text("node energy collection")
                        dpg.add_slider_float(
                            tag=tags.Slider.node_energy_collection_ratio,
                            default_value=0.9,
                            min_value=0,
                            max_value=1,
                            width=config.drag_width,
                            callback=callbacks.render_energy,
                            user_data=(
                                self.env_wrapper,
                                self.f,
                                self.energy_ax,
                                self,
                                self.fig_dpi,
                                True,
                            ),
                        )
                    with dpg.group():
                        dpg.add_text("energy backpropagation decay")
                        dpg.add_slider_float(
                            tag=tags.Slider.energy_backpropagate_decay,
                            default_value=0.1,
                            min_value=0,
                            max_value=1,
                            width=config.drag_width,
                            callback=callbacks.render_energy,
                            user_data=(
                                self.env_wrapper,
                                self.f,
                                self.energy_ax,
                                self,
                                self.fig_dpi,
                                True,
                            ),
                        )
                    with dpg.group():
                        dpg.add_text("node alive energy consumption")
                        dpg.add_drag_float(
                            tag=tags.Drag.node_alive_energy_consumption,
                            default_value=0.001,
                            min_value=0,
                            max_value=1,
                            speed=0.001,
                            width=config.drag_width,
                            callback=callbacks.render_energy,
                            user_data=(
                                self.env_wrapper,
                                self.f,
                                self.energy_ax,
                                self,
                                self.fig_dpi,
                                True,
                            ),
                        )

            with dpg.collapsing_header(tag="camera", label="camera", default_open=True):
                with dpg.group():
                    dpg.add_text("zoom")
                    dpg.add_drag_float(
                        tag=tags.Drag.camera_zoom,
                        min_value=0.0,
                        default_value=2,
                        speed=0.01,
                        width=config.drag_width,
                        callback=callbacks.render_all,
                        user_data=(
                            self.env_wrapper,
                            self.f,
                            self.tree_ax,
                            self.energy_ax,
                            self,
                            self.fig_dpi,
                            False,
                        ),
                    )
                with dpg.group():
                    dpg.add_text("elevation")
                    dpg.add_drag_float(
                        tag=tags.Drag.camera_elevation,
                        default_value=15,
                        speed=0.1,
                        width=config.drag_width,
                        callback=callbacks.render_all,
                        user_data=(
                            self.env_wrapper,
                            self.f,
                            self.tree_ax,
                            self.energy_ax,
                            self,
                            self.fig_dpi,
                            False,
                        ),
                    )
                with dpg.group():
                    dpg.add_text("azimuthal")
                    dpg.add_drag_float(
                        tag=tags.Drag.camera_azimuthal,
                        default_value=0,
                        speed=1,
                        min_value=-360,
                        max_value=360,
                        width=config.drag_width,
                        callback=callbacks.render_all,
                        user_data=(
                            self.env_wrapper,
                            self.f,
                            self.tree_ax,
                            self.energy_ax,
                            self,
                            self.fig_dpi,
                            False,
                        ),
                    )
            with dpg.collapsing_header(
                tag="operator", label="operator", default_open=True
            ):
                with dpg.group():
                    dpg.add_button(
                        tag=tags.Button.generate,
                        label="Generate",
                        callback=callbacks.render_all,
                        user_data=(
                            self.env_wrapper,
                            self.f,
                            self.tree_ax,
                            self.energy_ax,
                            self,
                            self.fig_dpi,
                            True,
                        ),
                    )

    def __create_render_window(self):
        # with dpg.window(
        #     tag=tags.Window.render,
        #     pos=[config.arbor_editor_width, 0],
        #     width=config.viewport_size[0] - config.arbor_editor_width,
        #     height=config.viewport_size[1],
        #     # no_title_bar=True,
        #     no_close=True,
        #     # no_collapse=True,
        #     no_move=True,
        #     no_resize=True,
        # ):
        #     dpg.add_image(
        #         texture_tag=tags.Texture.render,
        #         width=config.render_window_size[0],
        #         height=int(
        #             config.render_window_size[0] * self.buffer_wh[1] / self.buffer_wh[0]
        #         ),
        #     )

        with dpg.plot(
            label="WorkSpace",
            width=-1,
            height=int(config.viewport_size[1] * 0.9),
            pos=[config.arbor_editor_width, 0],
            equal_aspects=True,
        ):
            # optionally create legend
            dpg.add_plot_legend(parent=dpg.last_item())
            self.xaxis = dpg.add_plot_axis(
                dpg.mvXAxis,
                label="x axis",
                invert=False,
                parent="WorkSpace",
            )
            self.yaxis = dpg.add_plot_axis(
                dpg.mvYAxis,
                label="y axis",
                # invert=True,
                parent="WorkSpace",
            )
            print(self.buffer_wh)
            dpg.add_image_series(
                texture_tag=tags.Texture.render,
                bounds_min=[0, 0],
                bounds_max=self.buffer_wh,
                label="image",
                parent=self.yaxis,
                show=True,
            )

    def __del__(self):
        dpg.destroy_context()


if __name__ == "__main__":
    dearbor = Dearbor()
    dearbor.start()
# dpg.create_viewport(

#     title="Dearbor",
#     width=config.viewport_size[0],
#     height=config.viewport_size[1],
#     min_width=800,
#     min_height=800,
# )
# dpg.setup_dearpygui()
# dpg.show_viewport()
# dpg.set_primary_window(tags.Window.primary, True)
# dpg.start_dearpygui()
