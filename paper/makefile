all: fascd.pdf

_figures := cartoon
figures := $(addprefix genfigs/, $(addsuffix .pdf, $(_figures)))

cropfigures: $(figures)
	for X in $(figures); do \
	     (pdfcrop $$X $$X); \
	done

genfigs/cartoon.pdf: genfigs/cartoon.py
	(cd genfigs/ && python3 cartoon.py)

fascd.pdf: fascd.tex fascd.aux fascd.bbl 
	pdflatex fascd
	pdflatex fascd

fascd.aux: fascd.tex cropfigures
	pdflatex fascd
	bibtex fascd

fascd.bbl: fascd.aux fascd.bib
	bibtex fascd

# generate .zip of source files suitable for submitting to arxiv
publish.zip: fascd.pdf fascd.bbl
	rm -rf publish/
	mkdir publish/
	mkdir publish/figs/
	cp -f fixfigs/*.png fixfigs/vcycle/*.png publish/figs/
	cp -f genfigs/*.pdf publish/figs/
	mkdir publish/tikz/
	cp -f tikz/*.tex publish/tikz/
	cp -f siamart220329.cls siamplain.bst fascd.bib fascd.bbl fascd.tex publish/
	zip -r publish publish

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.blg *.snm *.toc *.nav *.vrb *.bbl *.synctex.gz *.thm *.dvi
	@rm -rf genfigs/*.pdf
	@rm -rf genfigs/__pycache__/
	@rm -rf publish/ publish.zip
