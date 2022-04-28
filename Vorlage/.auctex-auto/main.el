(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-run-style-hooks
    "header"
    "content/einleitung"
    "content/theorie"
    "content/durchfuehrung"
    "content/auswertung"
    "content/diskussion"
    "content/messwerte"))
 :latex)

