.PHONY: dev test lint process smoke

dev:
	uvicorn openphonic.main:app --reload

test:
	pytest

lint:
	ruff check .

process:
	openphonic process $(INPUT) --output $(OUTPUT)

smoke:
	openphonic smoke-test
