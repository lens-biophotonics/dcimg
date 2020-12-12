.PHONY: all wheel clean
all: wheel
wheel:
	python3 setup.py sdist bdist_wheel
clean:
	rm -fr build dist *.egg-info
test:
	python -m unittest discover
