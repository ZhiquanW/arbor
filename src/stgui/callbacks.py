import sys
import streamlit as st


import sim.torch_arbor as arbor

ARBOR_PROB_EPSILON = 0.001
ARBOR_ANGLE_EPSILON = 1


def max_steps_callback(self) -> None:
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.max_steps = int(
            st.session_state["max_steps"]
        )


def max_barnches_num_callbacks(self) -> None:
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.max_branches_num = int(
            st.session_state["max_branches_num"]
        )


def move_distance_range_callback():
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_distance_range = st.session_state["move_distance_range"]


def min_move_dir_offset_angle_x_callback():
    st.session_state["min_move_dir_offset_angle_x"] = min(
        st.session_state["min_move_dir_offset_angle_x"],
        st.session_state["max_move_dir_offset_angle_x"] - ARBOR_ANGLE_EPSILON,
    )
    st.session_state.move_dir_offset_angle_range_x = (
        st.session_state["min_move_dir_offset_angle_x"],
        st.session_state["max_move_dir_offset_angle_x"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_x = (
            st.session_state["min_move_dir_offset_angle_x"],
            st.session_state["max_move_dir_offset_angle_x"],
        )


def max_move_dir_offset_angle_x_callback():
    st.session_state["max_move_dir_offset_angle_x"] = max(
        st.session_state["min_move_dir_offset_angle_x"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_x"],
    )
    st.session_state["move_dir_offset_angle_range_x"] = (
        st.session_state["min_move_dir_offset_angle_x"],
        st.session_state["max_move_dir_offset_angle_x"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_x = (
            st.session_state["min_move_dir_offset_angle_x"],
            st.session_state["max_move_dir_offset_angle_x"],
        )


def min_move_dir_offset_angle_y_callback():
    st.session_state["min_move_dir_offset_angle_y"] = min(
        st.session_state["min_move_dir_offset_angle_y"],
        st.session_state["max_move_dir_offset_angle_y"] - ARBOR_ANGLE_EPSILON,
    )
    st.session_state["move_dir_offset_angle_range_y"] = (
        st.session_state["min_move_dir_offset_angle_y"],
        st.session_state["max_move_dir_offset_angle_y"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_y = (
            st.session_state["min_move_dir_offset_angle_y"],
            st.session_state["max_move_dir_offset_angle_y"],
        )


def max_move_dir_offset_angle_y_callback():
    st.session_state["max_move_dir_offset_angle_y"] = max(
        st.session_state["min_move_dir_offset_angle_y"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_y"],
    )

    st.session_state.move_dir_offset_angle_range_y = (
        st.session_state["min_move_dir_offset_angle_y"],
        st.session_state["max_move_dir_offset_angle_y"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_y = (
            st.session_state["min_move_dir_offset_angle_y"],
            st.session_state["max_move_dir_offset_angle_y"],
        )


def min_move_dir_offset_angle_z_callback():
    st.session_state["min_move_dir_offset_angle_z"] = min(
        st.session_state["min_move_dir_offset_angle_z"],
        st.session_state["max_move_dir_offset_angle_z"] - ARBOR_ANGLE_EPSILON,
    )
    st.session_state.move_dir_offset_angle_range_z = (
        st.session_state["min_move_dir_offset_angle_z"],
        st.session_state["max_move_dir_offset_angle_z"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_z = (
            st.session_state["min_move_dir_offset_angle_z"],
            st.session_state["max_move_dir_offset_angle_z"],
        )


def max_move_dir_offset_angle_z_callback():
    st.session_state["max_move_dir_offset_angle_z"] = max(
        st.session_state["min_move_dir_offset_angle_z"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_z"],
    )
    st.session_state.move_dir_offset_angle_range_z = (
        st.session_state["min_move_dir_offset_angle_z"],
        st.session_state["max_move_dir_offset_angle_z"],
    )
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.move_dir_offset_angle_range_z = (
            st.session_state["min_move_dir_offset_angle_z"],
            st.session_state["max_move_dir_offset_angle_z"],
        )


def branching_angle_range_callback():
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.branching_angle_range = (
            st.session_state["min_branching_angle"],
            st.session_state["max_branching_angle"],
        )


def branching_prob_range_callback():
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.branching_prob_range = (
            st.session_state["min_branching_prob"],
            st.session_state["max_branching_prob"],
        )


def min_sleep_prob_callback():
    st.session_state["min_sleep_prob"] = min(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"] - ARBOR_PROB_EPSILON,
    )
    min_sleep_prob = min(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"],
    )
    max_sleep_prob = max(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"],
    )
    st.session_state.node_sleep_prob_range = (
        min_sleep_prob,
        max_sleep_prob,
    )
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.node_sleep_prob_range = (
            min_sleep_prob,
            max_sleep_prob,
        )


def max_sleep_prob_callback():
    st.session_state["max_sleep_prob"] = max(
        st.session_state["min_sleep_prob"] + ARBOR_PROB_EPSILON,
        st.session_state["max_sleep_prob"],
    )
    min_sleep_prob = min(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"],
    )
    max_sleep_prob = max(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"],
    )
    st.session_state.node_sleep_prob_range = (
        min_sleep_prob,
        max_sleep_prob,
    )
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.node_sleep_prob_range = (
            min_sleep_prob,
            max_sleep_prob,
        )


def initial_energy_callback():
    if "env_wrapper" in st.session_state:
        st.session_state["env_wrapper"].env.arbor_engine.energy_engine = float(
            st.session_state["init_energy"]
        )


def max_energy_callback():
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.energy_module.max_energy = float(
            st.session_state["max_energy"]
        )


def collection_factor_callback():
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.energy_module.collection_factor = float(
            st.session_state["collection_factor"]
        )

        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.energy_module.recompute_energy_hist()


def node_consumption_factor_callback():
    if "env_wrapper" in st.session_state:
        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.energy_module.node_consumption_factor = float(
            st.session_state["node_consumption_factor"]
        )

        st.session_state[
            "env_wrapper"
        ].env.arbor_engine.energy_module.recompute_energy_hist()
