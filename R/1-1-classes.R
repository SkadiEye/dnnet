#################################
#### dnnetInput class
#' An S4 class containing predictors (x), response (y) and sample weights (w)
#'
#' @slot x A numeric matrix, the predictors
#' @slot y A factor or numeric vector, either the class labels or continuous responses
#' @slot w A numeric vector, sample weights
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' @export
setClass("dnnetInput",
         slots = list(
           x = "matrix",
           y = "ANY",
           w = "numeric"
         ))

#################################
#### dnnet class
#' An S4 class containing a deep neural network
#'
#' @slot norm A list, containing the centers and s.d.'s of x matrix and y vector (if numeric)
#' @slot weight A list of matrices, weight matrices in the fitted neural network
#' @slot bias A list of vectors, bias vectors in the fitted neural network
#' @slot loss The minimum loss acheived from the validate set
#' @slot label If the model is classification, a character vectors containing class labels
#' @slot model.type Either "Classification" or "Regression"
#' @slot model.spec Other possible information
#'
#' @seealso
#' \code{\link{dnnetInput-class}}\cr
#' @export
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
