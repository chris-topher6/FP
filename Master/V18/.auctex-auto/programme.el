(TeX-add-style-hook
 "programme"
 (lambda ()
   (LaTeX-add-bibitems
    "matplotlib"
    "numpy"
    "scipy"
    "uncertainties"
    "affinity"
    "iminuit"))
 '(or :bibtex :latex))

