# create conda environment as 
# conda create -n arbor python=3.9
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
trail-pc_tree:
	python trail/trail_pc_tree_env.py
