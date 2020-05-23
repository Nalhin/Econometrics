.PHONY: run latex format lint make-all

run:
	poetry run python -m src

latex:
	cd ./latex && xelatex -output-directory=../output root.tex

format:
	poetry run black src

lint:|
	poetry run flake8
	poetry run black src --check

all:
	make run && make latex
