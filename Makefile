.PHONY: install lint lint-fix test train run-api mlflow

install:
	pip install -e .[dev]

lint:
	ruff check .

lint-fix:
	ruff check . --fix

test:
	pytest -v

train:
	python scripts/train_mlp.py --data data/raw/telco_churn.csv --out models/

run-api:
	uvicorn src.api.main:app --reload

mlflow:
	mlflow ui