all: la_relazione.pdf

la_relazione.pdf: marticle.cls mstyle.sty rel.tex
	pdflatex rel.tex
	mv rel.pdf la_relazione.pdf

whole: marticle.cls mstyle.sty rel.tex
	pdflatex rel.tex
	pdflatex rel.tex
	pdflatex rel.tex
	mv rel.pdf la_relazione.pdf
