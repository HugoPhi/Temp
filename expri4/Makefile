# Makefile to compile LaTeX to PDF and clean up auxiliary files

# Name of the LaTeX source file (without extension)
LATEX_FILE = figure.tex

# Output PDF file name
PDF_FILE = figure.pdf

# Default target to generate the PDF
all: $(PDF_FILE)

# Rule to compile LaTeX to PDF
$(PDF_FILE): $(LATEX_FILE)
	pdflatex $(LATEX_FILE)
	pdflatex $(LATEX_FILE)  # Run twice for references

# Clean up auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg

# Phony targets (not actual files)
.PHONY: all clean
