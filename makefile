# create conda environment as 
# conda create -n arbor python=3.9
starbor:
	streamlit run trail/trail_starbor.py
dearbor:
	python src/gui/dearbor.py
# run the trail codes for CoreTreeEnv
trail-core-tree[torch]:
	@echo 'run trail-core-tree[torch]'
	python trail/trail_torch_arbor.py
trail-core-tree[local]:
	@echo 'run trail-core-tree[local]'
	python trail/trail_core_env.py
trail-core-tree:
	python trail/trail_core_env.py
env:
	pip install -r requirements.txt
test:
	python tests/utils_test.py
exp:
	python tests/exp.py
easy:
	python src/main.py
trail-collision:
	python trail/exp_collision.py
trail-core:
	python trail/trail_core.py
# run the trail codes for point cloud tree env
trail-pc_tree:
	python trail/trail_pc_tree_env.py
