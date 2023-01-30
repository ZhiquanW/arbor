env:
	pip install -r requirements.txt
test:
	python tests/utils_test.py
exp:
	python tests/exp.py
easy:
	python src/main.py
