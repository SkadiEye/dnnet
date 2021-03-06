###########################################################
### Define functions: importDnnet and splitDnnet

#' Import Data to create a \code{dnnetInput} object.
#'
#' @param x A \code{matrix} containing all samples/variables. It has to be \code{numeric}
#' and cannot be left blank. Any variable with missing value will be removed.
#' @param y A \code{numeric} or \code{factor} vector, indicating a continuous outcome or class label.
#' @param w A \code{numeric} vector, the sample weight. Will be 1 if left blank.
#'
#' @return An \code{dnnetInput} object.
#'
#' @importFrom methods new
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#' @export
importDnnet <- function(x, y, w = rep(1, length(y))) {

  new("dnnetInput", x=as.matrix(x), y=y, w=w)
}

#' A function to generate indice
#'
#' @param split As in \code{\link{dnnetInput-class}}.
#' @param n Sample size
#'
#' @return Returns a integer vector of indice.
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#'
#' @export
getSplitDnnet <- function(split, n) {

  if(is.numeric(split) && length(split) == 1 && split < 1)
    split <- sample(n, floor(n * split))

  if(is.numeric(split) && length(split) == 1 && split > 1)
    split <- 1:split

  if(is.character(split) && length(split) == 1 && split == "bootstrap")
    split <- sample(n, replace = TRUE)

  split
}

#' A function to split the \code{dnnetInput} object into a list of two \code{dnnetInput} objects:
#' one names train and the other named valid.
#'
#' @param object A \code{dnnetInput} object.
#' @param split A character, numeric variable or a numeric vector declaring a way to split
#' the \code{dnnetInput}. If it's number between 0 and 1, all samples will be split into two subsets
#' randomly, with the \code{train} containing such proportion of all samples and \code{valid} containing
#' the rest. If split is a character and is "bootstrap", the \code{train} will be a bootstrap sample
#' of the original data set and the \code{valid} will contain out-of-bag samples. If split is a vector
#' of integers, the \code{train} will contain samples whose indice are in the vector, and \code{valid} will
#' contain the rest.
#'
#' @return Returns a list of two \code{dnnetInput} objects.
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#'
#' @export
splitDnnet <-function(object, split) {

  split <- getSplitDnnet(split, dim(object@x)[1])

  train <- object
  train@x <- object@x[split, ]
  train@y <- object@y[split]
  train@w <- object@w[split]

  valid <- object
  valid@x <- object@x[-split, ]
  valid@y <- object@y[-split]
  valid@w <- object@w[-split]

  list(train = train, valid = valid)
}
