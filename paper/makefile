all: mcd2.pdf

mcd2.pdf: mcd2.tex mcd2.bib
	pdflatex mcd2
	bibtex mcd2
	pdflatex mcd2
	pdflatex mcd2

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz
