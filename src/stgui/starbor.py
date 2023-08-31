import os
import sys

import torch
import streamlit as st
import rlvortex
import rlvortex.envs.base_env as base_env

import stgui.callbacks as callbacks

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import tree_envs  # noqa: E402
import sim.torch_arbor as arbor  # noqa: E402
import sim.aux_space as aux_space
import sim.energy_module as energy

import render


class Starbor:
    def __init__(self) -> None:
        pass

    def launch(self):
        self.__init_gui()
        if "env_wrapper" not in st.session_state:  # type: ignore
            self.reset()

    def __init_gui(self):
        st.set_page_config(layout="wide")
        button_row0_cols = st.sidebar.columns(
            3,
        )
        with button_row0_cols[0]:
            if st.button("Reset"):
                self.reset()
        with button_row0_cols[1]:
            if st.button("Grow"):
                self.grow()
        with button_row0_cols[2]:
            if st.button("Cut"):
                self.cut()
        button_row1_cols = st.sidebar.columns(3)
        with button_row1_cols[0]:
            if st.button("Step"):
                self.step()
        with button_row1_cols[1]:
            if st.button("Stay"):
                self.stay()
        with button_row1_cols[2]:
            st.session_state.stay_step_num = st.number_input(
                "Stay Step Num", value=1, min_value=1, step=1
            )
        with st.sidebar:
            st.session_state.max_steps = st.number_input(
                "Max Grow Steps", value=20, min_value=1, step=1
            )
            st.session_state.max_branches_num = st.number_input(
                "Max Branches Number", value=50, min_value=1, step=1
            )
            with st.expander("Node Move Parameters", expanded=False):
                st.session_state.move_distance_range = st.slider(
                    "Move Distance Range",
                    value=(0.3, 0.5),
                    min_value=0.0,
                    max_value=1.0,
                )
                st.write("Move Direction Offset Angle Range")
                move_dir_offset_angle_range_x_cols = st.columns(2)
                min_move_dir_offset_angle_x = move_dir_offset_angle_range_x_cols[
                    0
                ].number_input(
                    "x axis : min",
                    key="min_move_dir_offset_angle_x",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )

                max_move_dir_offset_angle_x = move_dir_offset_angle_range_x_cols[
                    1
                ].number_input(
                    "x axis : max",
                    key="max_move_dir_offset_angle_x",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_x_callback,
                )
                st.session_state.move_dir_offset_angle_range_x = (
                    min_move_dir_offset_angle_x,
                    max_move_dir_offset_angle_x,
                )
                move_dir_offset_angle_range_y_cols = st.columns(2)
                min_move_dir_offset_angle_y = move_dir_offset_angle_range_y_cols[
                    0
                ].number_input(
                    "y axis : min",
                    key="min_move_dir_offset_angle_y",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )

                max_move_dir_offset_angle_y = move_dir_offset_angle_range_y_cols[
                    1
                ].number_input(
                    "y axis : max",
                    key="max_move_dir_offset_angle_y",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_y_callback,
                )
                st.session_state.move_dir_offset_angle_range_y = (
                    min_move_dir_offset_angle_y,
                    max_move_dir_offset_angle_y,
                )

                move_dir_offset_angle_range_z_cols = st.columns(2)
                min_move_dir_offset_angle_z = move_dir_offset_angle_range_z_cols[
                    0
                ].number_input(
                    "z axis : min",
                    key="min_move_dir_offset_angle_z",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )

                max_move_dir_offset_angle_z = move_dir_offset_angle_range_z_cols[
                    1
                ].number_input(
                    "z axis : max",
                    key="max_move_dir_offset_angle_z",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_z_callback,
                )
                st.session_state.move_dir_offset_angle_range_z = (
                    min_move_dir_offset_angle_z,
                    max_move_dir_offset_angle_z,
                )
            with st.expander("Node Branch Parameters", expanded=False):
                st.session_state.branching_angle_range = st.slider(
                    "Branching Angle Range", value=(0, 30), min_value=0, max_value=180
                )
                st.session_state.branching_prob_range = st.slider(
                    "Branching Probability Range",
                    value=(0.1, 0.5),
                    min_value=0.0,
                    max_value=1.0,
                )
            st.write("Node Sleep Probability Range")
            sleep_prob_range_cols = st.columns(2)
            min_sleep_prob = sleep_prob_range_cols[0].number_input(
                "min",
                key="min_sleep_prob",
                value=0.001,
                min_value=0.0,
                step=0.001,
                format="%.3f",
                on_change=callbacks.min_sleep_prob_callback,
            )
            max_sleep_prob = sleep_prob_range_cols[1].number_input(
                "max",
                key="max_sleep_prob",
                value=0.01,
                min_value=0.0,
                step=0.001,
                format="%.3f",
                on_change=callbacks.max_sleep_prob_callback,
            )
            min_sleep_prob = min(min_sleep_prob, max_sleep_prob)
            max_sleep_prob = max(min_sleep_prob, max_sleep_prob)
            st.session_state.node_sleep_prob_range = (min_sleep_prob, max_sleep_prob)
            with st.expander("Occupancy Space", expanded=True):
                st.session_state.occupancy_voxel_size = st.number_input(
                    "voxel size",
                    key="occupancy_space_voxel_size",
                    value=0.2,
                    min_value=0.01,
                    step=0.01,
                )
                st.session_state.occupancy_space_half_voxel_size = st.number_input(
                    "space half voxel size", value=50, min_value=1, step=1
                )
                st.write(
                    "space world size",
                    st.session_state.occupancy_voxel_size
                    * st.session_state.occupancy_space_half_voxel_size
                    * 2,
                )
            with st.expander("Shadow Space", expanded=True):
                st.session_state.shadow_voxel_size = st.number_input(
                    "voxel size",
                    value=0.2,
                    min_value=0.01,
                    step=0.01,
                )
                st.session_state.shadow_space_half_voxel_size = st.number_input(
                    "shadow space half voxel size", value=50, min_value=1, step=1
                )
                st.session_state.pyramid_half_size = st.number_input(
                    "pyramid half size", value=5, min_value=1, step=1
                )
                st.session_state.shadow_delta = st.number_input(
                    "shadow delta", value=0.05, min_value=0.01, step=0.01
                )
            with st.expander("Energy Space", expanded=True):
                st.session_state.init_energy = st.number_input(
                    "initial energy", value=10.0, min_value=0.0, step=1.0
                )
                st.session_state.max_energy = st.number_input(
                    "max energy", value=100.0, min_value=1.0, step=1.0
                )
                _ = st.number_input(
                    "colection voxel half size",
                    key="collection_voxel_half_size",
                    value=4,
                    min_value=1,
                    step=1,
                    on_change=callbacks.collection_voxel_half_size_callback,
                )

                st.session_state.collection_ratio = st.number_input(
                    "collection ratio",
                    value=0.8,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                )

                st.session_state.maintainence_consumption_factor = st.number_input(
                    "maintainence consumption", value=0.1, min_value=0.0, step=0.01
                )
                st.session_state.branch_consumption_factor = st.number_input(
                    "branch consumption", value=1.0, min_value=0.0, step=0.01
                )
                st.session_state.move_consumption_factor = st.number_input(
                    "move consumption", value=0.5, min_value=0.0, step=0.01
                )
        # if "energy_plot" in st.session_state:
        #     st.plotly_chart(st.session_state.energy_plot, use_container_width=True)
        if "tree_plot" in st.session_state:
            st.plotly_chart(st.session_state.tree_plot, use_container_width=True)

    def __init_engine(self) -> None:
        occupancy_space = aux_space.TorchOccupancySpace(
            voxel_size=st.session_state.occupancy_voxel_size,
            space_half_size=st.session_state.occupancy_space_half_voxel_size,
            device=torch.device("cpu"),
        )
        shadow_space = aux_space.TorchShadowSpace(
            voxel_size=st.session_state.shadow_voxel_size,
            space_half_size=st.session_state.shadow_space_half_voxel_size,
            pyramid_half_size=st.session_state.pyramid_half_size,
            shadow_delta=st.session_state.shadow_delta,
            device=torch.device("cpu"),
        )
        energy_module: energy.EnergyModule = energy.EnergyModule(
            init_energy=st.session_state.init_energy,
            max_energy=st.session_state.max_energy,
            collection_voxel_half_size=st.session_state.collection_voxel_half_size,
            init_collection_ratio=st.session_state.collection_ratio,
            maintainence_consumption_factor=st.session_state.maintainence_consumption_factor,
            move_consumption_factor=st.session_state.move_consumption_factor,
            branch_consumption_factor=st.session_state.branch_consumption_factor,
            record=True,
        )
        arbor_engine = arbor.TorchArborEngine(
            max_steps=int(st.session_state.max_steps),
            max_branches_num=int(st.session_state.max_branches_num),
            move_dis_range=[
                float(st.session_state.move_distance_range[0]),
                float(st.session_state.move_distance_range[1]),
            ],
            move_rot_range=[
                list(st.session_state.move_dir_offset_angle_range_x),
                list(st.session_state.move_dir_offset_angle_range_y),
                list(st.session_state.move_dir_offset_angle_range_z),
            ],
            new_branch_rot_range=list(st.session_state.branching_angle_range),
            node_branch_prob_range=list(st.session_state.branching_prob_range),
            node_sleep_prob_range=list(st.session_state.node_sleep_prob_range),
            occupancy_space=occupancy_space,
            shadow_space=shadow_space,
            energy_module=energy_module,
            device=torch.device("cpu"),
        )
        env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
            env=tree_envs.CoreTorchEnv(arbor_engine=arbor_engine)
        )
        env_wrapper.awake()
        env_wrapper.reset()
        st.session_state.env_wrapper = env_wrapper

    def grow(self) -> None:
        self.__init_engine()
        env_wrapper = st.session_state.env_wrapper
        while True:
            _, _, done, _ = env_wrapper.step(env_wrapper.sample_action())
            if done:
                break
        st.session_state.tree_plot = render.plotly_tree_skeleton(
            None, env_wrapper.env.arbor_engine
        )
        # st.session_state.energy_plot = render.plotly_energy_module(
        #     None, env_wrapper.env.arbor_engine.energy_module.energy_hist
        # )

    def reset(self) -> None:
        self.__init_engine()
        if "tree_plot" in st.session_state:
            del st.session_state["tree_plot"]

    def step(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        if env_wrapper.env.arbor_engine.done:
            return
        env_wrapper.step()
        st.session_state.tree_plot = render.plotly_tree_skeleton(
            None, env_wrapper.env.arbor_engine
        )
        st.session_state.env_wrapper = env_wrapper

    def cut(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        env_wrapper.env.arbor_engine.random_cut()
        st.session_state.env_wrapper = env_wrapper
        st.session_state.tree_plot = render.plotly_tree_skeleton(
            None, env_wrapper.env.arbor_engine, shadow=False, occupancy=False
        )

    def stay(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        if env_wrapper.env.arbor_engine.done:
            return
        for _ in range(st.session_state.stay_step_num):
            env_wrapper.env.arbor_engine.step_energy()
        st.session_state.tree_plot = render.plotly_tree_skeleton(
            None, env_wrapper.env.arbor_engine
        )
        # st.session_state.energy_plot = render.plotly_energy_module(
        #     None, env_wrapper.env.arbor_engine.energy_module.energy_hist
        # )
        st.session_state.env_wrapper = env_wrapper
