import rlvortex.envs.base_env as base_env
import rlvortex


def generate():
    arbor_engine = arbor.TorchArborEngine(
        max_steps=dpg.get_value(tags.Drag.max_steps),
        max_branches_num=dpg.get_value(tags.Drag.max_branches_num),
        move_dis_range=dpg.get_value(tags.Drag.move_distance_range)[:2],
        move_rot_range=[
            dpg.get_value(tags.Drag.move_rot_range_x)[:2],
            dpg.get_value(tags.Drag.move_rot_range_y)[:2],
            dpg.get_value(tags.Drag.move_rot_range_z)[:2],
        ],
        new_branch_rot_range=dpg.get_value(tags.Drag.new_branch_rot_range)[:2],
        node_branch_prob_range=dpg.get_value(tags.Drag.node_branch_prob_range)[:2],
        node_sleep_prob_range=dpg.get_value(tags.Drag.node_sleep_prob_range)[:2],
        device=torch.device("cpu"),
    )
    dearbor.env_wrapper: base_env.BaseEnvTrait = rlvortex.envs.base_env.EnvWrapper(
        env=tree_envs.CoreTorchEnv(arbor_engine=arbor_engine)
    )
    dearbor.env_wrapper.awake()
    dearbor.env_wrapper.reset()
