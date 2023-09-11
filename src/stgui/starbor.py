import os
import sys

import torch
import streamlit as st
import rlvortex
import rlvortex.envs.base_env as base_env

import stgui.callbacks as callbacks

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
import envs.tree_envs as tree_envs  # noqa: E402
import sim.torch_arbor as arbor  # noqa: E402
import sim.aux_space as aux_space
import sim.energy_module as energy

import utils.render as render
import train.trainer_params as trainer_params


class Starbor:
    def __init__(self) -> None:
        self.default_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nn_models",
            "trail_model.pth",
        )
        st.session_state.model = torch.load(self.default_model_path)

    def launch(self):
        self.__init_gui()
        if "env_wrapper" not in st.session_state:  # type: ignore
            self.__init_engine()

    def load_nn_model(self) -> None:
        pass

    def __init_gui(self):
        st.set_page_config(layout="wide")
        st.session_state.nn_model_checkbox = st.sidebar.checkbox("Enable NN Model Step")
        st.session_state.uploaded_model = st.file_uploader("Choose a model")
        if st.session_state.uploaded_model is not None:
            st.session_state.model = torch.load(st.session_state.uploaded_model)

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
            if st.button("Colect"):
                self.collect()
        with button_row1_cols[2]:
            st.session_state.stay_step_num = st.number_input(
                "Collect Step Num", value=1, min_value=1, step=1
            )
        with st.sidebar:
            st.number_input(
                "Max Grow Steps",
                value=20,
                min_value=1,
                step=1,
                on_change=callbacks.max_steps_callback,
                key="max_steps",
            )
            st.number_input(
                "Max Branches Number",
                value=50,
                min_value=1,
                step=1,
                on_change=callbacks.max_barnches_num_callbacks,
                key="max_branches_num",
            )
            with st.expander("Node Move Parameters", expanded=False):
                st.slider(
                    "Move Distance Range",
                    value=(0.3, 0.5),
                    min_value=0.0,
                    max_value=1.0,
                    on_change=callbacks.move_distance_range_callback,
                    key="move_distance_range",
                )
                st.write("Move Direction Offset Angle Range")
                move_dir_offset_angle_range_x_cols = st.columns(2)
                move_dir_offset_angle_range_x_cols[0].number_input(
                    "x axis : min",
                    key="min_move_dir_offset_angle_x",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )
                move_dir_offset_angle_range_x_cols[1].number_input(
                    "x axis : max",
                    key="max_move_dir_offset_angle_x",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_x_callback,
                )
                st.session_state["move_dir_offset_angle_range_x"] = (
                    st.session_state["min_move_dir_offset_angle_x"],
                    st.session_state["max_move_dir_offset_angle_x"],
                )
                move_dir_offset_angle_range_y_cols = st.columns(2)
                move_dir_offset_angle_range_y_cols[0].number_input(
                    "y axis : min",
                    key="min_move_dir_offset_angle_y",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )
                move_dir_offset_angle_range_y_cols[1].number_input(
                    "y axis : max",
                    key="max_move_dir_offset_angle_y",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_y_callback,
                )
                st.session_state["move_dir_offset_angle_range_y"] = (
                    st.session_state["min_move_dir_offset_angle_y"],
                    st.session_state["max_move_dir_offset_angle_y"],
                )

                move_dir_offset_angle_range_z_cols = st.columns(2)
                move_dir_offset_angle_range_z_cols[0].number_input(
                    "z axis : min",
                    key="min_move_dir_offset_angle_z",
                    value=-10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.min_move_dir_offset_angle_x_callback,
                )

                move_dir_offset_angle_range_z_cols[1].number_input(
                    "z axis : max",
                    key="max_move_dir_offset_angle_z",
                    value=10,
                    min_value=-180,
                    max_value=180,
                    on_change=callbacks.max_move_dir_offset_angle_z_callback,
                )
                st.session_state["move_dir_offset_angle_range_z"] = (
                    st.session_state["min_move_dir_offset_angle_z"],
                    st.session_state["max_move_dir_offset_angle_z"],
                )
            with st.expander("Node Branch Parameters", expanded=False):
                st.slider(
                    "Branching Angle Range",
                    value=(0, 30),
                    min_value=0,
                    max_value=180,
                    on_change=callbacks.branching_angle_range_callback,
                    key="branching_angle_range",
                )
                st.slider(
                    "Branching Probability Range",
                    value=(0.1, 0.5),
                    min_value=0.0,
                    max_value=1.0,
                    on_change=callbacks.branching_prob_range_callback,
                    key="branching_prob_range",
                )
                st.write("Node Sleep Probability Range")
                sleep_prob_range_cols = st.columns(2)
                sleep_prob_range_cols[0].number_input(
                    "min",
                    key="min_sleep_prob",
                    value=0.001,
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    on_change=callbacks.min_sleep_prob_callback,
                )
                sleep_prob_range_cols[1].number_input(
                    "max",
                    key="max_sleep_prob",
                    value=0.01,
                    min_value=0.0,
                    step=0.001,
                    format="%.3f",
                    on_change=callbacks.max_sleep_prob_callback,
                )

                st.session_state["node_sleep_prob_range"] = (
                    min(
                        st.session_state["min_sleep_prob"],
                        st.session_state["max_sleep_prob"],
                    ),
                    max(
                        st.session_state["min_sleep_prob"],
                        st.session_state["max_sleep_prob"],
                    ),
                )
            with st.expander("Energy Module", expanded=True):
                st.number_input(
                    "initial energy",
                    value=10.0,
                    min_value=0.0,
                    step=1.0,
                    on_change=callbacks.initial_energy_callback,
                    key="init_energy",
                )
                st.number_input(
                    "max energy",
                    value=100.0,
                    min_value=1.0,
                    step=1.0,
                    on_change=callbacks.max_energy_callback,
                    key="max_energy",
                )
                st.number_input(
                    "apical node collection ",
                    value=0.8,
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    on_change=callbacks.collection_factor_callback,
                    key="collection_factor",
                )
                st.number_input(
                    "every node consumption",
                    value=0.1,
                    min_value=0.0,
                    step=0.01,
                    on_change=callbacks.node_consumption_factor_callback,
                    key="node_consumption_factor",
                )

        self.__update_plot()
        if "tree_plot" in st.session_state:
            st.plotly_chart(st.session_state.tree_plot, use_container_width=True)

    def __init_engine(self) -> None:
        energy_module: energy.EnergyModule = energy.EnergyModule(
            init_energy=st.session_state["init_energy"],
            max_energy=st.session_state["max_energy"],
            collection_factor=st.session_state["collection_factor"],
            node_consumption_factor=st.session_state["node_consumption_factor"],
            record_history=True,
        )
        arbor_engine = arbor.TorchArborEngine(
            max_steps=int(st.session_state["max_steps"]),
            max_branches_num=int(st.session_state["max_branches_num"]),
            move_dis_range=st.session_state["move_distance_range"],
            move_rot_range=[
                list(st.session_state["move_dir_offset_angle_range_x"]),
                list(st.session_state["move_dir_offset_angle_range_y"]),
                list(st.session_state["move_dir_offset_angle_range_z"]),
            ],
            new_branch_rot_range=list(st.session_state.branching_angle_range),
            node_branch_prob_range=list(st.session_state.branching_prob_range),
            node_sleep_prob_range=list(st.session_state.node_sleep_prob_range),
            energy_module=energy_module,
            device=torch.device("cpu"),
        )
        if "env_wrapper" not in st.session_state:
            st.session_state["env_wrapper"] = rlvortex.envs.base_env.EnvWrapper(
                env=tree_envs.BranchProbArborEnv(arbor_engine=arbor_engine)
            )
            st.session_state["env_wrapper"].awake()
            st.session_state["obs"], _ = st.session_state["env_wrapper"].reset()

    def __update_plot(self) -> None:
        if "env_wrapper" in st.session_state:
            env_wrapper = st.session_state.env_wrapper
            st.session_state.tree_plot = render.plotly_tree_skeleton(
                None, env_wrapper.env.arbor_engine
            )
            st.session_state.tree_plot = render.plotly_energy_module(
                st.session_state.tree_plot,
                env_wrapper.env.arbor_engine.energy_module.energy_hist,
            )

    def grow(self) -> None:
        st.session_state.env_wrapper.reset()
        env_wrapper = st.session_state.env_wrapper
        o, _ = env_wrapper.reset()
        while True:
            action = (
                st.session_state.model.act(st.session_state.obs)
                if st.session_state.nn_model_checkbox
                else env_wrapper.sample_action()
            )
            st.session_state.obs, _, done, _ = env_wrapper.step(action)
            if done:
                break
        self.__update_plot()

    def reset(self) -> None:
        st.session_state.env_wrapper.reset()

    def step(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        if env_wrapper.env.arbor_engine.done:
            return
        action = (
            st.session_state.model.act(st.session_state.obs)
            if st.session_state.nn_model_checkbox
            else env_wrapper.sample_action()
        )
        st.session_state.obs, _, done, _ = env_wrapper.step(action)
        st.session_state.env_wrapper = env_wrapper
        print(st.session_state.obs)
        self.__update_plot()

    def cut(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        env_wrapper.env.arbor_engine.random_cut()
        st.session_state.env_wrapper = env_wrapper
        self.__update_plot()

    def collect(self) -> None:
        env_wrapper = st.session_state.env_wrapper
        if env_wrapper.env.arbor_engine.done:
            return
        for _ in range(st.session_state.stay_step_num):
            env_wrapper.env.arbor_engine.step_energy()
        st.session_state.env_wrapper = env_wrapper
        self.__update_plot()
