VENV := /tmp/bw-nlp-venv
PY   := $(VENV)/bin/python
UV   := $(VENV)/bin/uvicorn

.PHONY: setup train run docker-build docker-run

setup:
	@bash setup.sh

train:
	$(PY) src/train.py

run:
	$(UV) src.main:app --reload --host 0.0.0.0 --port 8001

docker-build:
	sudo docker build -t bayarwoy-nlp .

docker-run:
	sudo docker run -p 8001:8001 --name bayarwoy-nlp bayarwoy-nlp