install:
	pip install -e .[dev]

lint:
	ruff check .

lint-fix:
	ruff check . --fix

test:
	pytest -v

run-api:
	uvicorn src.api.main:app --reload

mlflow:
	mlflow ui