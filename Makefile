.PHONY: all wheel clean
all: wheel
wheel:
	python3 -m build
clean:
	rm -fr build dist *.egg-info
test:
	python -m unittest discover
