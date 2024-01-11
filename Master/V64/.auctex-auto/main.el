(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-run-style-hooks
    "header"
    "content/1_introduction"
    "content/2_theory"
    "content/3_procedure"
    "content/4_analysis"
    "content/5_conclusion"))
 :latex)

