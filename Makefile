USER := $(shell whoami)
ifeq ($(USER), Yam)
	PY = python3.8
else
	PY = python
endif

.PHONY: test
test:
	$(PY) -m pytest tests/

.PHONY: build
build:
	$(PY) setup.py sdist bdist_wheel

.PHONY: upload
upload:
	$(PY) -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
	@rm -rf build dist *.egg-info

.PHONY: clean
clean:
	@rm -rf build dist *.egg-info
