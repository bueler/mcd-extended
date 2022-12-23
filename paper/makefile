all: fascd.pdf

_figures := cartoon
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/cartoon.pdf: genfigs/cartoon.py
	(cd genfigs/ && python3 cartoon.py)

fascd.pdf: fascd.tex fascd.bib cropfigures
	pdflatex fascd
	bibtex fascd
	pdflatex fascd
	pdflatex fascd

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
	@rm -rf genfigs/*.pdf
	@rm -rf genfigs/__pycache__/
