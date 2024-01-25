# Variablen
LATEX_COMPILER = lualatex
BIB_COMPILER = biber
PYTHON = python
LATEX_FLAGS = -output-directory=build
BIB_FLAGS = --output-directory build

# Standardziel
all: build/main.pdf

# PDF erstellen
build/main.pdf: main.tex build/main.bbl build/main.blg | build
	$(LATEX_COMPILER) $(LATEX_FLAGS) main.tex
	$(LATEX_COMPILER) $(LATEX_FLAGS) main.tex

# Bibliographie erstellen
build/main.bbl build/main.blg: main.bcf | build
	$(BIB_COMPILER) $(BIB_FLAGS) main

# Erste LaTeX-Kompilierung für Bibliographie
main.bcf: main.tex content/*.tex build/plots | build
	$(LATEX_COMPILER) $(LATEX_FLAGS) main.tex

# Python-Skripte ausführen und Plots erstellen
build/plots: script/*.py data/* | build
	for script in script/*.py; do \
		$(PYTHON) "$$script"; \
	done

# Ordner erstellen
build:
	mkdir -p build

# Aufräumen
clean:
	rm -rf build

# Hilfsziel
.PHONY: all clean
