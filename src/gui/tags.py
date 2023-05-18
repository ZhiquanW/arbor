primary_window = "primary_window"


class Window:
    primary = "window/primary"
    operator = "window/operator"
    render = "window/render"


class Drag:
    max_steps = "drag_float/max_steps"
    max_branches_num = "drag_float/max_braches_num"
    move_distance_range = "drag_floatx/move_distance_range"
    move_rot_range_x = "drag_floatx/move_rot_range_x"
    move_rot_range_y = "drag_floatx/move_rot_range_y"
    move_rot_range_z = "drag_floatx/move_rot_range_z"
    new_branch_rot_range = "drag_floatx/new_branch_rot_range"
    node_branch_prob_range = "drag_floatx/node_branch_prob_range"
    node_sleep_prob_range = "drag_floatx/node_sleep_prob_range"
    node_alive_energy_consumption = "drag_float/energy_alive_consumption"
    camera_zoom = "drag_float/camera_zoom"
    camera_elevation = "drag_float/camera_elevation"
    camera_azimuthal = "drag_float/camera_azimuthal"


class Slider:
    node_energy_collection_ratio = "slider_float/node_energy_collection"
    energy_backpropagate_decay = "slider_float/energy_backpropagate_decay"


class Button:
    generate = "button/generate"
    collect = "button/collect"


class CollapsingHeader:
    arbor_editor = "collapsing_header/arbor_editor"
    energy = "collapsing_header/energy"


class Texture:
    render = "texture/render"
