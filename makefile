env:
	pip install -r requirements.txt
test:
	python tests/utils_test.py
exp:
	python tests/exp.py
easya:
	python src/main.py
