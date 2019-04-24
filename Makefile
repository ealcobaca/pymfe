all: clean code_check test

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.md5" -o -name "*.pyd" -o -name "*~" | xargs rm -f
	rm -rf coverage
	rm -rf dist
	rm -rf build
	rm -rf docs/_build
	rm -rf docs/source/generated

test:
	rm -rf coverage .coverage
	pytest tests/ --showlocals -v --cov=pymfe/

code_check:
	flake8 pymfe | grep -v __init__
	pylint pymfe -d 'C0103, R0913, R0902, R0914, C0302, R0904, R0801, E1101'
	mypy pymfe --ignore-missing-imports
