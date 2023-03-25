PY = python

.PHONY: test
test:
	$(PY) -m pytest tests/

.PHONY: build
build:
	python setup.py sdist bdist_wheel

.PHONY: upload
upload:
	python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
	@rm -rf build dist *.egg-info

.PHONY: clean
clean:
	@rm -rf ./dist/ ./build/ ./pnlp.egg-info/