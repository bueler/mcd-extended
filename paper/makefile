all: mcd2.pdf

_figures := cartoon
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/cartoon.pdf: genfigs/cartoon.py
	(cd genfigs/ && python3 cartoon.py)

mcd2.pdf: mcd2.tex mcd2.bib cropfigures
	pdflatex mcd2
	bibtex mcd2
	pdflatex mcd2
	pdflatex mcd2

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
