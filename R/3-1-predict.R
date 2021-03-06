###########################################################
### Predict using dnnet objects

#' @describeIn Predict new observations using \code{dnnet} object.
#'
#' @param object A \code{dnnet} object.
#' @param newData A matrix with the same number of columns in the input data.
#' @param type Consistent with model.type in the \code{object}.
#'
#' @return A numeric vector for regression or a matrix of probabilities for each class for classification.
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#'
#' @export
setMethod("predict",
          "dnnet",
          function(object, newData, type, ...) {

            n.layer <- length(object@bias) - 1
            activate <- get(object@model.spec$activate)
            one_sample_size <- rep(1, dim(newData)[1])
            newData <- (newData - outer(rep(1, dim(newData)[1]), object@norm$x.center)) /
              outer(rep(1, dim(newData)[1]), object@norm$x.scale)
            for(j in 1:n.layer) {

              if(j == 1) {
                pred <- activate(newData %*% object@weight[[j]] + one_sample_size %*% object@bias[[j]])
              } else {
                pred <- activate(pred %*% object@weight[[j]] + one_sample_size %*% object@bias[[j]])
              }
            }
            pred <- (pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]])[, 1]
            if(object@model.type == "classification") {
              pred <- 1/(exp(-pred) + 1)
              return(matrix(cbind(pred, 1-pred), dim(newData)[1], length(object@label),
                            dimnames = list(NULL, object@label)))
            }

            return(pred*object@norm$y.scale + object@norm$y.center)
          })
