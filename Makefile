.PHONY: run latex format lint make-all test

run:
	poetry run python -m src

latex:
	cd ./latex && xelatex -output-directory=../output root.tex

format:
	poetry run black src

test:
	poetry run coverage run -m unittest discover

test_coverage:
	poetry run coverage report -m

lint:|
	poetry run flake8
	poetry run black src --check

all:
	make run && make latex
