# quant-harness — common dev tasks. Run from the quant-harness/ directory.
.PHONY: help install test example audit-example api docker calibrate calibrate-quick clean

help:
	@echo "install         editable install with dev + api extras (pip install -e .[dev,api])"
	@echo "test            run the pytest suite"
	@echo "example         run the data-free quickstart demo"
	@echo "audit-example   run the synthetic qh-audit demo (writes examples/audit_sample_returns.csv)"
	@echo "api             serve the verdict API locally on :8000 (uvicorn)"
	@echo "docker          build the verdict API image qh-api:dev"
	@echo "calibrate       regenerate reports/CALIBRATION.{json,md} (200 seeds, ~minutes)"
	@echo "calibrate-quick fast calibration (40 seeds, no PBO)"
	@echo "clean           remove caches and build artifacts"

install:
	pip install -e ".[dev,api]"

test:
	python -m pytest

example:
	python examples/quickstart.py

audit-example:
	python examples/audit_quickstart.py

api:
	python -m uvicorn harness.service.app:app --host 0.0.0.0 --port 8000 --no-access-log

docker:
	docker build -t qh-api:dev .

calibrate:
	python run_calibration.py --seeds 200

calibrate-quick:
	python run_calibration.py --quick --out CALIBRATION_quick

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
