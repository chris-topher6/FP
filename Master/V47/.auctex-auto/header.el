(TeX-add-style-hook
 "header"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrartcl" "bibliography=totoc" "captions=tableheading" "titlepage=firstiscover" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("rerunfilecheck" "aux") ("babel" "english") ("unicode-math" "math-style=ISO" "bold-style=ISO" "sans-style=italic" "nabla=upright" "partial=upright" "warnings-off={           % ┐
    mathtools-colon,       % │ unnötige Warnungen ausschalten
    mathtools-overbracket, % │
  }" "") ("siunitx" "separate-uncertainty=true" "per-mode=symbol-or-fraction" "") ("mhchem" "version=4" "math-greek=default" "text-greek=default" "") ("csquotes" "autostyle") ("placeins" "section" "below" "") ("caption" "labelfont=bf" "font=small" "width=0.9\\textwidth" "") ("biblatex" "backend=biber" "sorting=none" "") ("hyperref" "german" "unicode" "pdfusetitle" "pdfcreator={}" "pdfproducer={}" "") ("extdash" "shortcuts")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "scrartcl"
    "scrartcl10"
    "scrhack"
    "rerunfilecheck"
    "amsmath"
    "amssymb"
    "mathtools"
    "fontspec"
    "babel"
    "unicode-math"
    "siunitx"
    "mhchem"
    "csquotes"
    "xfrac"
    "float"
    "placeins"
    "pdflscape"
    "caption"
    "subcaption"
    "graphicx"
    "grffile"
    "booktabs"
    "adjustbox"
    "longtable"
    "microtype"
    "biblatex"
    "hyperref"
    "bookmark"
    "extdash")
   (TeX-add-symbols
    '("ket" ["argument"] 1)
    '("bra" ["argument"] 1))
   (LaTeX-add-bibliographies
    "lit"
    "programme"))
 :latex)

