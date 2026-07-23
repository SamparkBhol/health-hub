UV ?= uv
PY := $(UV) run
WEB := apps/web

.PHONY: install models doctor assets refresh-public-health refresh-seasonal train-public-outlook seed lint typecheck test postgres-smoke web-build verify api demo clean

install:
	$(UV) sync --all-groups --all-extras
	npm --prefix $(WEB) ci

models:
	$(PY) python scripts/fetch_models.py

doctor:
	$(PY) python scripts/doctor.py

assets:
	$(PY) python scripts/build_boundaries.py
	$(PY) python scripts/audit_epiclim.py

refresh-public-health:
	$(PY) python scripts/collect_public_health.py

refresh-seasonal:
	$(PY) python scripts/collect_seasonal.py

train-public-outlook:
	$(PY) python scripts/train_public_outlook.py

seed:
	$(PY) python scripts/make_synthetic.py

lint:
	$(PY) ruff check packages services workers pipelines scripts tests

typecheck:
	$(PY) mypy --explicit-package-bases packages services workers pipelines scripts

test:
	$(PY) pytest

postgres-smoke:
	$(PY) python scripts/postgres_smoke.py

web-build:
	mkdir -p $(WEB)/public/data
	cp data/boundaries/odisha_districts_census_2011.geojson $(WEB)/public/data/odisha_districts_census_2011.geojson
	npm --prefix $(WEB) run build

verify: doctor seed lint typecheck test web-build
	ACCEPTANCE_PREREQUISITES=doctor,seed,ruff,mypy,pytest,web-build $(PY) python scripts/write_acceptance_report.py

api:
	$(PY) uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env

demo:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov apps/web/dist data/synthetic
