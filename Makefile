.PHONY: test lint format check

test:
	pytest -q
	python -m py_compile maze_solver/*.py config.py train.py evaluate.py

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

check: lint test

train:
	python train.py --episodes 5000 --seed 0

# lint alias
