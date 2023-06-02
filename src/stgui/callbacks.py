import sys
import streamlit as st


ARBOR_PROB_EPSILON = 0.001
ARBOR_ANGLE_EPSILON = 1


def min_sleep_prob_callback():
    st.session_state["min_sleep_prob"] = min(
        st.session_state["min_sleep_prob"],
        st.session_state["max_sleep_prob"] - ARBOR_PROB_EPSILON,
    )


def max_sleep_prob_callback():
    st.session_state["max_sleep_prob"] = max(
        st.session_state["min_sleep_prob"] + ARBOR_PROB_EPSILON,
        st.session_state["max_sleep_prob"],
    )


def min_move_dir_offset_angle_x_callback():
    st.session_state["min_move_dir_offset_angle_x"] = min(
        st.session_state["min_move_dir_offset_angle_x"],
        st.session_state["max_move_dir_offset_angle_x"] - ARBOR_ANGLE_EPSILON,
    )


def max_move_dir_offset_angle_x_callback():
    st.session_state["max_move_dir_offset_angle_x"] = max(
        st.session_state["min_move_dir_offset_angle_x"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_x"],
    )


def min_move_dir_offset_angle_y_callback():
    st.session_state["min_move_dir_offset_angle_y"] = min(
        st.session_state["min_move_dir_offset_angle_y"],
        st.session_state["max_move_dir_offset_angle_y"] - ARBOR_ANGLE_EPSILON,
    )


def max_move_dir_offset_angle_y_callback():
    st.session_state["max_move_dir_offset_angle_y"] = max(
        st.session_state["min_move_dir_offset_angle_y"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_y"],
    )


def min_move_dir_offset_angle_z_callback():
    st.session_state["min_move_dir_offset_angle_z"] = min(
        st.session_state["min_move_dir_offset_angle_z"],
        st.session_state["max_move_dir_offset_angle_z"] - ARBOR_ANGLE_EPSILON,
    )


def max_move_dir_offset_angle_z_callback():
    st.session_state["max_move_dir_offset_angle_z"] = max(
        st.session_state["min_move_dir_offset_angle_z"] + ARBOR_ANGLE_EPSILON,
        st.session_state["max_move_dir_offset_angle_z"],
    )
