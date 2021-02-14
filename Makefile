PACKAGE := pymfe
TEST_NCORES := auto

all: clean install-dev code-check test-cov
.PHONY: all clean test test-cov code-check pypi install install-dev html help h t c cl

clean: ## Clean all undesired files such as .so, .pyc, build files and etc.
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf .coverage.*
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	rm -rf docs/source/generated
	rm -rf docs/source/auto_examples
	cd docs ; make clean_all

cl: clean ## Shortcut to clean

test: ## Execute the code test using pytest.
	pytest -n $(TEST_NCORES) tests/

test-cov: ## Execute the code test using pytest and measuring the coverage.
	rm -rf coverage .coverage
	pytest -n $(TEST_NCORES) --cov=$(PACKAGE)/ tests/

t: test-cov ## Shortcut to test-cov

code-check: ## Execute the code check with flake8, pylint, mypy.
	flake8 $(PACKAGE)
	pylint $(PACKAGE) -j 0 -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101, C0330, E1136'
	mypy $(PACKAGE) --ignore-missing-imports

type-check:  ## Execute the code check with mypy only.
	mypy $(PACKAGE) --ignore-missing-imports

c: code-check # Shortcut to code-check

pypi: clean ## Send the package to pypi.
	pip install -U twine wheel
	python3 setup.py sdist bdist_wheel
	twine upload dist/*

install-dev: ## Install pymfe for developers using pip.
	pip install -U -e .
	pip install -U -r requirements.txt
	pip install -U -r requirements-dev.txt
	pip install -U -r requirements-docs.txt

install: ## Install the package using pip.
	pip install .

html: ## Create the online documentation.
	cd docs; make html

help: ## List target command description.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

h: help ## Shortcut to help

format: ## format all the package using black
	@black --line-length 79 pymfe/
