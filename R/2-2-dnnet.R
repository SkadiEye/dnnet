###########################################################
### Multilayer Perceptron Model for Regression or Classification

#' Multilayer Perceptron Model for Regression or Classification
#'
#' Fit a Multilayer Perceptron Model for Regression or Classification
#'
#' @param train A \code{dnnetInput} object, the training set.
#' @param validate A \code{dnnetInput} object, the validation set, optional.
#' @param norm.x A boolean variable indicating whether to normalize the input matrix.
#' @param norm.y A boolean variable indicating whether to normalize the response (if continuous).
#' @param activate Activation Function. One of the following,
#'  "sigmoid", "tanh", "relu", "prelu", "elu", "celu".
#' @param learning.rate Initial learning rate, 0.001 by default; If "adam" is chosen as
#'  an adaptive learning rate adjustment method, 0.1 by defalut.
#' @param l1.reg weight for l1 regularization, optional.
#' @param l2.reg weight for l2 regularization, optional.
#' @param n.batch Batch size for batch gradient descent.
#' @param n.epoch Maximum number of epochs.
#' @param early.stop Indicate whether early stop is used (only if there exists a validation set).
#' @param early.stop.det Number of epochs of increasing loss to determine the early stop.
#' @param plot Indicate whether to plot the loss.
#' @param accel "rcpp" to use the Rcpp version and "none" (default) to use the R version for back propagation.
#' @param learning.rate.adaptive Adaptive learning rate adjustment methods, one of the following,
#'  "constant", "adadelta", "adagrad", "momentum", "adam".
#' @param epsilon A parameter used in Adagrad and Adam.
#' @param beta1 A parameter used in Adam.
#' @param beta2 A parameter used in Adam.
#' @param loss.f Loss function of choice.
#'
#' @return Returns a \code{DnnModelObj} object.
#'
#' @importFrom stats runif
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' \code{\link{dnnetInput-class}}\cr
#' \code{\link{actF}}
#'
#' @export
dnnet <- function(train, validate = NULL,
                  norm.x = TRUE, norm.y = ifelse(is.factor(train@y), FALSE, TRUE),
                  activate = "elu", n.hidden = c(10, 10),
                  learning.rate = ifelse(learning.rate.adaptive %in% c("adam"), 0.001, 0.01),
                  l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                  early.stop = ifelse(is.null(validate), FALSE, TRUE), early.stop.det = 5,
                  plot = FALSE, accel = c("rcpp", "gpu", "none")[3],
                  learning.rate.adaptive = c("constant", "adadelta", "adagrad", "momentum", "adam")[2],
                  rho = c(0.9, 0.95, 0.99, 0.999)[ifelse(learning.rate.adaptive == "momentum", 1, 3)],
                  epsilon = c(10**-10, 10**-8, 10**-6, 10**-4)[2],
                  beta1 = 0.9, beta2 = 0.999, loss.f = ifelse(is.factor(train@y), "logit", "mse"), ...) {

  if(!class(train@x) %in% c("matrix", "data.frame"))
    stop("x has to be either a matrix or a data frame. ")
  if(!class(train@y) %in% c("numeric", "factor", "vector", "integer"))
    stop("y has to be either a factor or a numeric vector. ")
  if(dim(train@x)[1] != length(train@y))
    stop("Dimensions of x and y do not match. ")

  learning.rate
  norm.y
  rho
  loss.f

  sample.size <- length(train@y)
  n.variable <- dim(train@x)[2]

  if(!is.null(train@w)) {

    if(length(train@w) != length(train@y))
      stop("Dimensions of y and w do not match. ")
    if(!class(train@w) %in% c("integer", "numeric"))
      stop("w has to be a numeric vector. ")
    if(sum(train@w < 0) > 0)
      stop("w has be to non-negative. ")
  } else {

    train@w <- rep(1, sample.size)
  }

  train@x <- as.matrix(train@x)
  norm <- list(x.center = rep(0, n.variable),
               x.scale = rep(1, n.variable),
               y.center = 0, y.scale = 1)
  if(norm.x && (sum(apply(train@x, 2, sd) == 0) == 0)) {

    train@x <- scale(train@x)
    norm$x.center <- attr(train@x, "scaled:center")
    norm$x.scale <- attr(train@x, "scaled:scale")

    if(!is.null(validate))
      validate@x <- (validate@x - outer(rep(1, length(validate@y)), norm$x.center))/outer(rep(1, length(validate@y)), norm$x.scale)
  }

  if(is.factor(train@y)) {

    label <- levels(train@y)
    train@y <- (train@y == label[1])*1

    if(!is.null(validate))
      validate@y <- (validate@y == label[1])*1

    model.type <- "classification"
  } else {

    if(norm.y) {

      train@y <- scale(train@y)
      norm$y.center <- attr(train@y, "scaled:center")
      norm$y.scale <- attr(train@y, "scaled:scale")

      if(!is.null(validate))
        validate@y <- (validate@y - norm$y.center)/norm$y.scale
    }
    model.type <- "regression"
  }

  if(sum(is.na(train@x)) > 0 | sum(is.na(train@y)) > 0)
    stop("Please remove NA's in the input data first. ")

  # if(identical(activate, "tanh")) {
  #   w.ini = 1
  # } else {
    w.ini = 0.1
  # }
    # if(is.factor(train@y)) w.ini = 1

  if(accel == "gpu") {

    if(!is.null(validate)) {

      try(result <- dnnet.backprop.gpu(n.hidden, w.ini,
                                       train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                                       get(activate), get(paste(activate, "_", sep = '')),
                                       n.epoch, n.batch, model.type,
                                       learning.rate, l1.reg, l2.reg, early.stop.det))
    } else {

      try(result <- dnnet.backprop.gpu(n.hidden, w.ini,
                                       train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                       get(activate), get(paste(activate, "_", sep = '')),
                                       n.epoch, n.batch, model.type,
                                       learning.rate, l1.reg, l2.reg, early.stop.det))
    }
  } else if(accel == "rcpp") {

    # learning.rate.adaptive.char <-
    #   c("c", "d", "g", "m", "a")[match(learning.rate.adaptive,
    #                                   c("constant", "adadelta", "adagrad", "momentum", "adam"))]

    if(!is.null(validate)) {

      try(result <- backprop(n.hidden, w.ini,
                             train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                             activate,
                             n.epoch, n.batch, model.type,
                             learning.rate, l1.reg, l2.reg, early.stop.det,
                             learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    } else {

      try(result <- backprop(n.hidden, w.ini,
                             train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                             activate,
                             n.epoch, n.batch, model.type,
                             learning.rate, l1.reg, l2.reg, early.stop.det,
                             learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    }
  } else {

    if(!is.null(validate)) {
      try(result <- dnnet.backprop.r(n.hidden, w.ini,
                                     train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    } else {

      try(result <- dnnet.backprop.r(n.hidden, w.ini,
                                     train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    }
  }

  if(!is.null(validate) & plot) try(plot(result[[3]][0:result[[4]]+1]*norm$y.scale**2, ylab = "loss"))
  if(is.na(result[[3]][1]) | is.nan(result[[3]][1])) {

    min.loss <- Inf
  } else {

    min.loss <- min(result[[3]][0:result[[4]]+1])*norm$y.scale**2
  }

  if(exists("result")) {

    return(methods::new("dnnet", norm = norm,
                        weight = result[[1]],
                        bias = result[[2]],
                        loss = min.loss,
                        label = ifelse(model.type == "regression", '', list(label))[[1]],
                        model.type = model.type,
                        model.spec = list(n.hidden = n.hidden,
                                          activate = activate,
                                          learning.rate = learning.rate,
                                          l1.reg = l1.reg,
                                          l2.reg = l2.reg,
                                          n.batch = n.batch,
                                          n.epoch = n.epoch)))
  } else {

    stop("Error fitting model. ")
  }
}

#' Back Propagation using gpuR (nor working)
NULL
dnnet.backprop.gpu <- function(n.hidden, w.ini,
                               x, y, w, valid, x.valid, y.valid, w.valid,
                               activate, activate_, n.epoch, n.batch, model.type,
                               learning.rate, l1.reg, l2.reg, early.stop.det) {

  # x_ <- gpuR::vclMatrix(x, type = "double")
  # y_ <- gpuR::vclVector(y, type = "double")
  # w_ <- gpuR::vclVector(w, type = "double")

  if(valid) {

    x_valid_ <- gpuR::vclMatrix(x.valid, type = "double")
    # y_valid_ <- gpuR::vclVector(y.valid, type = "double")
    # w_valid_ <- gpuR::vclVector(w.valid, type = "double")
    valid_sample_size <- gpuR::vclMatrix(rep(1, nrow(x.valid)), nrow(x.valid), 1, type = "double")
  }

  loss <- numeric(0)
  weight <- list()
  bias <- list()
  a <- list()
  h <- list()
  d_a <- list()
  d_h <- list()
  d_w <- list()

  best.loss <- Inf
  best.weight <- list()
  best.bias <- list()

  n.layer <- length(n.hidden)
  n.variable <- ncol(x)
  sample.size <- nrow(x)

  for(i in 1:(n.layer + 1)) {

    if(i == 1) {

      weight[[i]] <- vclMatrix(matrix(runif(n.variable * n.hidden[i], -1, 1), n.variable, n.hidden[1]) * w.ini, type = "double")
      bias[[i]] <- vclMatrix(matrix(runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2, type = "double")
    } else if(i == n.layer + 1) {

      weight[[i]] <- vclMatrix(matrix(runif(n.hidden[i-1] * 1, -1, 1), n.hidden[i-1], 1) * w.ini, type = "double")
      bias[[i]] <- vclMatrix(matrix(runif(1, -1, 1), 1, 1) * w.ini / 2, type = "double")
    } else {

      weight[[i]] <- vclMatrix(matrix(runif(n.hidden[i-1] * n.hidden[i], -1, 1), n.hidden[i-1], n.hidden[i]) * w.ini, type = "double")
      bias[[i]] <- vclMatrix(matrix(runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2, type = "double")
    }
  }

  n.round <- ceiling(sample.size / n.batch)
  i.bgn <- integer(n.round)
  i.end <- integer(n.round)
  n.s <- integer(n.round)
  one_sample_size <- list()
  one_sample_size_t <- list()
  for(s in 1:n.round) {

    i.bgn[s] <- (s-1)*n.batch + 1
    i.end[s] <- min(s*n.batch, sample.size)
    n.s[s] <- i.end[s] - i.bgn[s] + 1
    one_sample_size[[s]] <- vclMatrix(rep(1, n.s[s]), n.s[s], 1, type = "double")
    one_sample_size_t[[s]] <- gpuR::t(one_sample_size[[s]])
  }

  for(k in 1:n.epoch) {

    new.order <- sample(sample.size)
    x_ <- gpuR::vclMatrix(x[new.order, ], type = "double")
    y_ <- gpuR::vclMatrix(y[new.order], ncol = 1, nrow = sample.size, type = "double")
    w_ <- gpuR::vclMatrix(w[new.order], ncol = 1, nrow = sample.size, type = "double")

    for(i in 1:n.round) {

      xi <- block(x_, as.integer(i.bgn[i]), as.integer(i.end[i]), as.integer(1), as.integer(n.variable))
      yi <- block(y_, as.integer(i.bgn[i]), as.integer(i.end[i]), as.integer(1), as.integer(1))
      wi <- block(w_, as.integer(i.bgn[i]), as.integer(i.end[i]), as.integer(1), as.integer(1))

      for(j in 1:n.layer) {

        if(j == 1) {

          a[[j]] <- (xi %*% weight[[j]]) + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        } else {

          a[[j]] <- h[[j-1]] %*% weight[[j]] + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        }
      }

      y.pi <- h[[n.layer]] %*% weight[[n.layer + 1]] + one_sample_size[[i]] %*% bias[[n.layer + 1]]
      if(model.type == "classification")
        y.pi <- (1 + exp(-y.pi))**(-1)

      d_a[[n.layer + 1]] <- -(yi - y.pi) * wi / sum(wi)
      d_w[[n.layer + 1]] <- gpuR::t(h[[n.layer]]) %*% d_a[[n.layer + 1]]
      if(l1.reg > 0)
        weight[[n.layer + 1]] <- weight[[n.layer + 1]] - l1.reg*((weight[[n.layer + 1]][] > 0) - (weight[[n.layer + 1]][] < 0))
      if(l2.reg > 0)
        weight[[n.layer + 1]] <- (1 - l2.reg)*weight[[n.layer + 1]]
      weight[[n.layer + 1]] <- weight[[n.layer + 1]] - (learning.rate) * d_w[[n.layer + 1]]
      bias[[n.layer + 1]] <- bias[[n.layer + 1]] - (learning.rate) * (gpuR::t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])
      for(j in n.layer:1) {

        d_h[[j]] <- d_a[[j + 1]] %*% gpuR::t(weight[[j + 1]])
        d_a[[j]] <- d_h[[j]] * activate_(a[[j]])

        if(j > 1) {
          d_w[[j]] <- gpuR::t(h[[j - 1]]) %*% d_a[[j]]
        } else {
          d_w[[j]] <- gpuR::t(xi) %*% d_a[[j]]
        }
        if(l1.reg > 0)
          weight[[j]] <- weight[[j]] - l1.reg*((weight[[j]][] > 0) - (weight[[j]][] < 0))
        if(l2.reg > 0)
          weight[[j]] <- (1 - l2.reg)*weight[[j]]
        weight[[j]] <- weight[[j]] - (learning.rate) * d_w[[j]]
        bias[[j]] <- bias[[j]] - (learning.rate) * (one_sample_size_t[[i]] %*% d_a[[j]])
      }
    }

    if(valid) {

      for(j in 1:n.layer) {

        if(j == 1) {
          pred <- activate(x_valid_ %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        } else {
          pred <- activate(pred %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        }
      }
      pred <- (pred %*% weight[[n.layer + 1]] + valid_sample_size %*% bias[[n.layer + 1]])[, 1]
      if(model.type == "classification") {
        pred <- (exp(-pred) + 1)**(-1)
        loss[k] <- -sum(w.valid * (y.valid * log(pred) + (1-y.valid) * log(1-pred))) / sum(w.valid)
      } else {
        loss[k] <- sum(w.valid * (y.valid - pred)**2) / sum(w.valid)
      }

      if(is.na(loss[k]) | is.null(loss[k]) | is.nan(loss[k]) | is.infinite(loss[k])) {

        break
      } else {

        if(loss[k] < best.loss) {

          best.loss <- loss[k]
          for(j in 1:(n.layer + 1)) {
            best.weight[[j]] <- deepcopy(weight[[j]])
            best.bias[[j]] <- deepcopy(bias[[j]])
          }
        }

        if(k > early.stop.det + 1) {
          if(prod(loss[k:(k - early.stop.det + 1)] > loss[(k-1):(k - early.stop.det)]) > 0) {
            break
          }
        }
      }
    }
  }

  best_weight_ <- list()
  best_bias_ <- list()
  if(valid) {

    for(i in 1:(n.layer + 1)) {

      best_weight_[[j]] <- best.weight[[j]][]
      best_bias_[[j]] <- best.bias[[j]][]
    }
  } else {

    for(i in 1:(n.layer + 1)) {

      best_weight_[[j]] <- weight[[j]][]
      best_bias_[[j]] <- bias[[j]][]
    }
  }

  return(list(best_weight_, best_bias_, loss, length(loss)-1))
}

#' Back Propagation
NULL
dnnet.backprop.r <- function(n.hidden, w.ini,
                             x, y, w, valid, x.valid, y.valid, w.valid,
                             activate, activate_, n.epoch, n.batch, model.type,
                             learning.rate, l1.reg, l2.reg, early.stop.det,
                             learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f) {

  if(valid) {

    valid_sample_size <- matrix(rep(1, nrow(x.valid)), nrow(x.valid), 1)
  }

  loss <- numeric(0)
  weight <- list()
  bias <- list()
  a <- list()
  h <- list()
  d_a <- list()
  d_h <- list()
  d_w <- list()

  best.loss <- Inf
  best.weight <- list()
  best.bias <- list()

  n.layer <- length(n.hidden)
  n.variable <- ncol(x)
  sample.size <- nrow(x)

  for(i in 1:(n.layer + 1)) {

    if(i == 1) {

      weight[[i]] <- matrix(stats::runif(n.variable * n.hidden[i], -1, 1), n.variable, n.hidden[i]) * w.ini
      bias[[i]]   <- matrix(stats::runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2
    } else if(i == n.layer + 1) {

      weight[[i]] <- matrix(stats::runif(n.hidden[i-1] * 1, -1, 1), n.hidden[i-1], 1) * w.ini
      bias[[i]]   <- matrix(stats::runif(1, -1, 1), 1, 1) * w.ini / 2
    } else {

      weight[[i]] <- matrix(stats::runif(n.hidden[i-1] * n.hidden[i], -1, 1), n.hidden[i-1], n.hidden[i]) * w.ini
      bias[[i]]   <- matrix(stats::runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2
    }
  }

  best.weight <- weight
  best.bias <- bias

  dw <- list()
  db <- list()

  if(learning.rate.adaptive == "momentum") {

    last.dw <- list()
    last.db <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        last.dw[[i]] <- matrix(0, n.variable, n.hidden[i])
        last.db[[i]] <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        last.dw[[i]] <- matrix(0, n.hidden[i-1], 1)
        last.db[[i]] <- matrix(0, 1, 1)
      } else {

        last.dw[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        last.db[[i]] <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adagrad") {

    weight.ss <- list()
    bias.ss <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        weight.ss[[i]] <- matrix(0, n.variable, n.hidden[i])
        bias.ss[[i]]   <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        weight.ss[[i]] <- matrix(0, n.hidden[i-1], 1)
        bias.ss[[i]]   <- matrix(0, 1, 1)
      } else {

        weight.ss[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.ss[[i]]   <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adadelta") {

    weight.egs <- list()
    weight.es <- list()
    bias.egs <- list()
    bias.es <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        weight.egs[[i]] <- matrix(0, n.variable, n.hidden[i])
        bias.egs[[i]]   <- matrix(0, 1, n.hidden[i])
        weight.es[[i]]  <- matrix(0, n.variable, n.hidden[i])
        bias.es[[i]]    <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        weight.egs[[i]] <- matrix(0, n.hidden[i-1], 1)
        bias.egs[[i]]   <- matrix(0, 1, 1)
        weight.es[[i]]  <- matrix(0, n.hidden[i-1], 1)
        bias.es[[i]]    <- matrix(0, 1, 1)
      } else {

        weight.egs[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.egs[[i]]   <- matrix(0, 1, n.hidden[i])
        weight.es[[i]]  <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.es[[i]]    <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adam") {

    mt.w <- list()
    vt.w <- list()
    mt.b <- list()
    vt.b <- list()
    mt.ind <- 0
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        mt.w[[i]] <- matrix(0, n.variable, n.hidden[i])
        mt.b[[i]] <- matrix(0, 1, n.hidden[i])
        vt.w[[i]] <- matrix(0, n.variable, n.hidden[i])
        vt.b[[i]] <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        mt.w[[i]] <- matrix(0, n.hidden[i-1], 1)
        mt.b[[i]] <- matrix(0, 1, 1)
        vt.w[[i]] <- matrix(0, n.hidden[i-1], 1)
        vt.b[[i]] <- matrix(0, 1, 1)
      } else {

        mt.w[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        mt.b[[i]] <- matrix(0, 1, n.hidden[i])
        vt.w[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        vt.b[[i]] <- matrix(0, 1, n.hidden[i])
      }
    }
  }

  n.round <- ceiling(sample.size / n.batch)
  i.bgn <- integer(n.round)
  i.end <- integer(n.round)
  n.s <- integer(n.round)
  one_sample_size <- list()
  # one_sample_size_t <- list()
  for(s in 1:n.round) {

    i.bgn[s] <- (s-1)*n.batch + 1
    i.end[s] <- min(s*n.batch, sample.size)
    n.s[s] <- i.end[s] - i.bgn[s] + 1
    one_sample_size[[s]] <- matrix(rep(1, n.s[s]), n.s[s], 1)
    # one_sample_size_t[[s]] <- gpuR::t(one_sample_size[[s]])
  }

  for(k in 1:n.epoch) {

    new.order <- sample(sample.size)
    x_ <- x[new.order, ]
    y_ <- y[new.order]
    w_ <- w[new.order]

    for(i in 1:n.round) {

      xi <- x_[i.bgn[i]:i.end[i], ]
      yi <- y_[i.bgn[i]:i.end[i]]
      wi <- w_[i.bgn[i]:i.end[i]]

      for(j in 1:n.layer) {

        if(j == 1) {

          a[[j]] <- (xi %*% weight[[j]]) + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        } else {

          a[[j]] <- h[[j-1]] %*% weight[[j]] + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        }
      }

      y.pi <- h[[n.layer]] %*% weight[[n.layer + 1]] + one_sample_size[[i]] %*% bias[[n.layer + 1]]
      # if(model.type == "classification")
      if(loss.f == "logit")
        y.pi <- 1/(1 + exp(-y.pi))
      if(loss.f == "rmsle") {
        y.pi <- relu(y.pi)
        d_a[[n.layer + 1]] <- -(log(yi+1) - log(y.pi+1)) / (y.pi+1) * (y.pi > 0) * wi / sum(wi)
      } else {
        d_a[[n.layer + 1]] <- -(yi - y.pi) * wi / sum(wi)
      }

      d_w[[n.layer + 1]] <- t(h[[n.layer]]) %*% d_a[[n.layer + 1]]
      bias.grad <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])
      if(learning.rate.adaptive == "momentum") {

        last.dw[[n.layer + 1]] <- last.dw[[n.layer + 1]] * rho + d_w[[n.layer + 1]] * learning.rate
        last.db[[n.layer + 1]] <- last.db[[n.layer + 1]] * rho + bias.grad * learning.rate
        dw[[n.layer + 1]] <- last.dw[[n.layer + 1]]
        db[[n.layer + 1]] <- last.db[[n.layer + 1]]
      } else if (learning.rate.adaptive == "adagrad") {

        weight.ss[[n.layer + 1]] <- weight.ss[[n.layer + 1]] + d_w[[n.layer + 1]]**2
        bias.ss[[n.layer + 1]]   <- bias.ss[[n.layer + 1]]   + bias.grad**2
        dw[[n.layer + 1]] <- d_w[[n.layer + 1]]/sqrt(weight.ss[[n.layer + 1]] + epsilon) * learning.rate
        db[[n.layer + 1]] <- bias.grad/sqrt(bias.ss[[n.layer + 1]] + epsilon) * learning.rate
      } else if (learning.rate.adaptive == "adadelta") {

        weight.egs[[n.layer + 1]] <- weight.egs[[n.layer + 1]] * rho + (1-rho) * d_w[[n.layer + 1]]**2
        bias.egs[[n.layer + 1]]   <-   bias.egs[[n.layer + 1]] * rho + (1-rho) * bias.grad**2
        dw[[n.layer + 1]] <- sqrt(weight.es[[n.layer + 1]] + epsilon) /
          sqrt(weight.egs[[n.layer + 1]] + epsilon) * d_w[[n.layer + 1]]
        db[[n.layer + 1]] <- sqrt(bias.es[[n.layer + 1]] + epsilon) /
          sqrt(bias.egs[[n.layer + 1]] + epsilon) * bias.grad
        weight.es[[n.layer + 1]] <- weight.es[[n.layer + 1]] * rho + (1-rho) * dw[[n.layer + 1]]**2
        bias.es[[n.layer + 1]]   <-   bias.es[[n.layer + 1]] * rho + (1-rho) * db[[n.layer + 1]]**2
      } else if (learning.rate.adaptive == "adam") {

        mt.ind <- mt.ind + 1
        mt.w[[n.layer + 1]] <- mt.w[[n.layer + 1]] * beta1 + (1-beta1) * d_w[[n.layer + 1]]
        mt.b[[n.layer + 1]] <- mt.b[[n.layer + 1]] * beta1 + (1-beta1) * bias.grad
        vt.w[[n.layer + 1]] <- vt.w[[n.layer + 1]] * beta2 + (1-beta2) * d_w[[n.layer + 1]]**2
        vt.b[[n.layer + 1]] <- vt.b[[n.layer + 1]] * beta2 + (1-beta2) * bias.grad**2
        dw[[n.layer + 1]] <- learning.rate * mt.w[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.w[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
        db[[n.layer + 1]] <- learning.rate * mt.b[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.b[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
      } else {

        dw[[n.layer + 1]] <- d_w[[n.layer + 1]] * learning.rate
        db[[n.layer + 1]] <- bias.grad * learning.rate
      }
      weight[[n.layer + 1]] <- weight[[n.layer + 1]] - dw[[n.layer + 1]] -
        l1.reg*((weight[[n.layer + 1]] > 0) - (weight[[n.layer + 1]] < 0)) -
        l2.reg*weight[[n.layer + 1]]
      bias[[n.layer + 1]] <- bias[[n.layer + 1]] - db[[n.layer + 1]]
      for(j in n.layer:1) {

        d_h[[j]] <- d_a[[j + 1]] %*% t(weight[[j + 1]])
        d_a[[j]] <- d_h[[j]] * activate_(a[[j]])

        if(j > 1) {
          d_w[[j]] <- t(h[[j - 1]]) %*% d_a[[j]]
        } else {
          d_w[[j]] <- t(xi) %*% d_a[[j]]
        }
        # weight[[j]] <- weight[[j]] - learning.rate * d_w[[j]] -
        #   l1.reg*((weight[[j]] > 0) - (weight[[j]] < 0)) -
        #   l2.reg*weight[[j]]
        # bias[[j]] <- bias[[j]] - learning.rate * (t(one_sample_size[[i]]) %*% d_a[[j]])
        bias.grad <- (t(one_sample_size[[i]]) %*% d_a[[j]])
        if(learning.rate.adaptive == "momentum") {

          last.dw[[j]] <- last.dw[[j]] * rho + d_w[[j]] * learning.rate
          last.db[[j]] <- last.db[[j]] * rho + bias.grad * learning.rate
          dw[[j]] <- last.dw[[j]]
          db[[j]] <- last.db[[j]]
        } else if (learning.rate.adaptive == "adagrad") {

          weight.ss[[j]] <- weight.ss[[j]] + d_w[[j]]**2
          bias.ss[[j]]   <- bias.ss[[j]]   + bias.grad**2
          dw[[j]] <- d_w[[j]]/sqrt(weight.ss[[j]] + epsilon) * learning.rate
          db[[j]] <- bias.grad/sqrt(bias.ss[[j]] + epsilon) * learning.rate
        } else if (learning.rate.adaptive == "adadelta") {

          weight.egs[[j]] <- weight.egs[[j]] * rho + (1-rho) * d_w[[j]]**2
          bias.egs[[j]]   <-   bias.egs[[j]] * rho + (1-rho) * bias.grad**2
          dw[[j]] <- sqrt(weight.es[[j]] + epsilon) / sqrt(weight.egs[[j]] + epsilon) * d_w[[j]]
          db[[j]] <- sqrt(  bias.es[[j]] + epsilon) / sqrt(  bias.egs[[j]] + epsilon) * bias.grad
          weight.es[[j]] <- weight.es[[j]] * rho + (1-rho) * dw[[j]]**2
          bias.es[[j]]   <-   bias.es[[j]] * rho + (1-rho) * db[[j]]**2
        } else if (learning.rate.adaptive == "adam") {

          # mt.ind <- mt.ind + 1
          mt.w[[j]] <- mt.w[[j]] * beta1 + (1-beta1) * d_w[[j]]
          mt.b[[j]] <- mt.b[[j]] * beta1 + (1-beta1) * bias.grad
          vt.w[[j]] <- vt.w[[j]] * beta2 + (1-beta2) * d_w[[j]]**2
          vt.b[[j]] <- vt.b[[j]] * beta2 + (1-beta2) * bias.grad**2
          dw[[j]] <- learning.rate * mt.w[[j]] / (1-beta1**mt.ind) / (sqrt(vt.w[[j]] / (1-beta2**mt.ind)) + epsilon)
          db[[j]] <- learning.rate * mt.b[[j]] / (1-beta1**mt.ind) / (sqrt(vt.b[[j]] / (1-beta2**mt.ind)) + epsilon)
        } else {

          dw[[j]] <- d_w[[j]] * learning.rate
          db[[j]] <- bias.grad * learning.rate
        }
        # browser()
        weight[[j]] <- weight[[j]] - dw[[j]] - l1.reg*((weight[[j]] > 0) - (weight[[j]] < 0)) - l2.reg*weight[[j]]
        bias[[j]]   <- bias[[j]]   - db[[j]]
      }
    }

    if(valid) {

      for(j in 1:n.layer) {

        if(j == 1) {
          pred <- activate(x.valid %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        } else {
          pred <- activate(pred %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        }
      }
      pred <- (pred %*% weight[[n.layer + 1]] + valid_sample_size %*% bias[[n.layer + 1]])[, 1]
      # if(model.type == "classification") {
      if(loss.f == "logit") {
        pred <- 1/(exp(-pred) + 1)
        loss[k] <- -sum(w.valid * (y.valid * log(pred) + (1-y.valid) * log(1-pred))) / sum(w.valid)
      } else if(loss.f == "mse") {
        loss[k] <- sum(w.valid * (y.valid - pred)**2) / sum(w.valid)
      } else if(loss.f == "rmsle") {
        pred <- relu(pred)
        loss[k] <- sum(w.valid * (log(y.valid+1) - log(pred+1))**2) / sum(w.valid)
      }

      if(is.na(loss[k]) | is.null(loss[k]) | is.nan(loss[k]) | is.infinite(loss[k])) {

        loss <- loss[-k]
        break
      } else {

        if(loss[k] < best.loss) {

          best.loss <- loss[k]
          best.weight <- weight
          best.bias <- bias
        }

        if(k > early.stop.det + 1) {
          if(prod(loss[k:(k - early.stop.det + 1)] > loss[(k-1):(k - early.stop.det)]) > 0) {
            break
          }
        }
      }
    }
  }

  if(valid) {

    best_weight_ <- best.weight
    best_bias_ <- best.bias
  } else {

    best_weight_ <- weight
    best_bias_ <- bias
  }

  return(list(best_weight_, best_bias_, loss, length(loss)-1))
}








