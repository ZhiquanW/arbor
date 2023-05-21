import streamlit as st
import numpy as np
import pandas as pd


with st.sidebar:
    max_steps_input = st.number_input("Max Grow Steps", value=20, min_value=1, step=1)
    max_branches_num = st.number_input(
        "Max Branches Number", value=50, min_value=1, step=1
    )
    with st.expander("Node Move Parameters", expanded=True):
        move_distance_range = st.slider(
            "Move Distance Range", value=(0.3, 0.5), min_value=0.0, max_value=1.0
        )
        st.write("Move Direction Offset Angle Range")
        move_dir_offset_angle_range_x = st.slider(
            "x axis", value=(-30, 30), min_value=-180, max_value=180
        )
        move_dir_offset_angle_range_y = st.slider(
            "y axis", value=(-30, 30), min_value=-180, max_value=180
        )
        move_dir_offset_angle_range_z = st.slider(
            "z axis", value=(-30, 30), min_value=-180, max_value=180
        )

    with st.expander("Node Branch Parameters", expanded=True):
        branching_angle_range = st.slider(
            "Branching Angle Range", value=(30, 60), min_value=0, max_value=180
        )
        branching_prob_range = st.slider(
            "Branching Probability Range",
            value=(0.1, 0.5),
            min_value=0.0,
            max_value=1.0,
        )
    node_sleep_prob_range = st.slider(
        "Node Sleep Probability Range", value=(0.0, 0.5), min_value=0.0, max_value=1.0
    )

    if st.button("Generate"):
        st.write("Generating...")
        st.write("Done!")
