.PHONY: dev test lint process

dev:
	uvicorn openphonic.main:app --reload

test:
	pytest

lint:
	ruff check .

process:
	openphonic process $(INPUT) --output $(OUTPUT)
