# Template for Informatics UG final-year projects

Please base your project report on `skeleton.tex`, reading the instructions in
that example file carefully.

To compile the `skeleton.pdf` report, with all cross-references resolved:
```
pdflatex skeleton.tex
bibtex skeleton.aux
pdflatex skeleton.tex
pdflatex skeleton.tex
```

Many TeX distributions have (or can install) a `latexmk` command that will
automatically compile everything that is needed:
```
latexmk -pdf skeleton.tex
```

If the logo causes compilation problems (errors related to `eushield`), it isn't
necessary, you may remove the `logo` option from the first line of code.
Although check first that you are using `pdflatex` or the `-pdf` option above.
As directed in `skeleton.tex` do not change other template or layout options.

Occassionally latex gets really confused by errors, and intermediate files need
to be deleted before the report will compile again. We strongly recommend that
you keep your files in version control so that you can unpick any problems.
Remember also to keep off-site backups.
