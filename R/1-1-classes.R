setClass("dnnetInput",
         slots = list(
           x = "matrix",
           y = "ANY",
           w = "numeric"
         ))

setClass("dnnet",
         slots = list(
           norm = "list",
           weight = "list",
           bias = "list",
           loss = "numeric",
           label = "character",
           model.type = "character",
           model.spec = "list"
         ))
