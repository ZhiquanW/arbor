# create conda environment as 
# conda create -n arbor python=3.9
.ONESHELL:

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

starbor:
	streamlit run trail/trail_starbor.py --server.port=8080
env:
	$(CONDA_ACTIVATE) arbor
	pip install -r requirements.txt
test:
	python tests/utils_test.py
exp:
	python tests/exp.py
easy:
	python src/main.py

conda-activate:
	$(CONDA_ACTIVATE) arbor
trail-bp-tree: 
	@echo "activate conda env"
	$(CONDA_ACTIVATE) arbor
	@echo "start trail-bp-tree"
	python trail/trail_branch_prob_arbor_env.py
	@echo "trail-bp-tree done"
bp-trainer:
	@echo "activate conda env"
	$(CONDA_ACTIVATE) arbor
	@echo "start trail-bp-tree"
	python src/train/bp_trainer.py
	@echo "trail-bp-tree done"
pc:
	@echo "activate conda env"
	$(CONDA_ACTIVATE) arbor
	python src/point_cloud/pc_main.py
	@echo "point-cloud done"