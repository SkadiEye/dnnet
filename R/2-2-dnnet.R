dnnet <- function(train, validate = NULL,
                  norm.x = TRUE, norm.y = ifelse(is.factor(y), FALSE, TRUE),
                  activate = "elu", n.hidden = c(10, 10),
                  learning.rate = ifelse(learning.rate.adaptive %in% c("adam"), 0.001, 0.01),
                  l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                  early.stop = ifelse(is.null(validate), FALSE, TRUE), early.stop.det = 5,
                  plot = FALSE, accel = c("rcpp", "gpu", "none")[3],
                  learning.rate.adaptive = c("constant", "adadelta", "adagrad", "momentum", "adam")[2],
                  rho = c(0.9, 0.95, 0.99, 0.999)[3],
                  epsilon = c(10**-10, 10**-8, 10**-6, 10**-4)[2],
                  beta1 = 0.9, beta2 = 0.999, ...) {

  if(!class(train@x) %in% c("matrix", "data.frame"))
    stop("x has to be either a matrix or a data frame. ")
  if(!class(train@y) %in% c("numeric", "factor", "vector", "integer"))
    stop("y has to be either a factor or a numeric vector. ")
  if(dim(train@x)[1] != length(train@y))
    stop("Dimensions of x and y do not match. ")

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
  if(norm.x) {

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

    if(!is.null(validate)) {

      try(result <- backprop(n.hidden, w.ini,
                             train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                             get(activate), get(paste(activate, "_", sep = '')),
                             n.epoch, n.batch, strsplit(model.type, split = '')[[1]][1],
                             learning.rate, l1.reg, l2.reg, early.stop.det))
    } else {

      try(result <- backprop(n.hidden, w.ini,
                             train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                             get(activate), get(paste(activate, "_", sep = '')),
                             n.epoch, n.batch, strsplit(model.type, split = '')[[1]][1],
                             learning.rate, l1.reg, l2.reg, early.stop.det))
    }
  } else {

    if(!is.null(validate)) {
      try(result <- dnnet.backprop.r(n.hidden, w.ini,
                                     train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2))
    } else {

      try(result <- dnnet.backprop.r(n.hidden, w.ini,
                                     train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2))
    }
  }

  if(plot) try(plot(result[[3]][0:result[[4]]+1], ylab = "loss"))

  if(exists("result")) {

    return(methods::new("dnnet", norm = norm,
                        weight = result[[1]],
                        bias = result[[2]],
                        loss = min(result[[3]][0:result[[4]]+1]),
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

dnnet.backprop.r <- function(n.hidden, w.ini,
                             x, y, w, valid, x.valid, y.valid, w.valid,
                             activate, activate_, n.epoch, n.batch, model.type,
                             learning.rate, l1.reg, l2.reg, early.stop.det,
                             learning.rate.adaptive, rho, epsilon, beta1, beta2) {

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

        weight.ss[[i]] <- matrix(1, n.variable, n.hidden[i])
        bias.ss[[i]]   <- matrix(1, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        weight.ss[[i]] <- matrix(1, n.hidden[i-1], 1)
        bias.ss[[i]]   <- matrix(1, 1, 1)
      } else {

        weight.ss[[i]] <- matrix(1, n.hidden[i-1], n.hidden[i])
        bias.ss[[i]]   <- matrix(1, 1, n.hidden[i])
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
      if(model.type == "classification")
        y.pi <- 1/(1 + exp(-y.pi))

      d_a[[n.layer + 1]] <- -(yi - y.pi) * wi / sum(wi)
      d_w[[n.layer + 1]] <- t(h[[n.layer]]) %*% d_a[[n.layer + 1]]
      if(learning.rate.adaptive == "momentum") {

        if(i == 1 & k == 1) {

          dw[[n.layer + 1]] <- d_w[[n.layer + 1]] * learning.rate
          db[[n.layer + 1]] <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]]) * learning.rate
        } else {

          dw[[n.layer + 1]] <- d_w[[n.layer + 1]] * learning.rate +
            rho * last.dw[[n.layer + 1]]
          db[[n.layer + 1]] <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]]) * learning.rate +
            rho * last.db[[n.layer + 1]]
        }
      } else if (learning.rate.adaptive == "constant") {

        dw[[n.layer + 1]] <- d_w[[n.layer + 1]] * learning.rate
        db[[n.layer + 1]] <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]]) * learning.rate
      } else if (learning.rate.adaptive == "adagrad") {

        dw[[n.layer + 1]] <- d_w[[n.layer + 1]]/sqrt(weight.ss[[n.layer + 1]] + epsilon) * learning.rate
        db[[n.layer + 1]] <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])/sqrt(bias.ss[[n.layer + 1]] + epsilon) * learning.rate
      } else if (learning.rate.adaptive == "adadelta") {

        bias.grad <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])
        weight.egs[[n.layer + 1]] <- weight.egs[[n.layer + 1]] * rho + (1-rho) * d_w[[n.layer + 1]]**2
        bias.egs[[n.layer + 1]]   <-   bias.egs[[n.layer + 1]] * rho + (1-rho) * bias.grad**2
        dw[[n.layer + 1]] <- sqrt(weight.es[[n.layer + 1]] + epsilon) /
          sqrt(weight.egs[[n.layer + 1]] + epsilon) * d_w[[n.layer + 1]]
        db[[n.layer + 1]] <- sqrt(bias.es[[n.layer + 1]] + epsilon) /
          sqrt(bias.egs[[n.layer + 1]] + epsilon) * bias.grad
        weight.es[[n.layer + 1]] <- weight.es[[n.layer + 1]] * rho + (1-rho) * dw[[n.layer + 1]]**2
        bias.es[[n.layer + 1]]   <-   bias.es[[n.layer + 1]] * rho + (1-rho) * db[[n.layer + 1]]**2
      } else if (learning.rate.adaptive == "adam") {

        bias.grad <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])
        mt.ind <- mt.ind + 1
        mt.w[[n.layer + 1]] <- mt.w[[n.layer + 1]] * beta1 + (1-beta1) * d_w[[n.layer + 1]]
        mt.b[[n.layer + 1]] <- mt.b[[n.layer + 1]] * beta1 + (1-beta1) * bias.grad
        vt.w[[n.layer + 1]] <- vt.w[[n.layer + 1]] * beta2 + (1-beta2) * d_w[[n.layer + 1]]**2
        vt.b[[n.layer + 1]] <- vt.b[[n.layer + 1]] * beta2 + (1-beta2) * bias.grad**2
        dw[[n.layer + 1]] <- learning.rate * mt.w[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.w[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
        db[[n.layer + 1]] <- learning.rate * mt.b[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.b[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
      }
      weight[[n.layer + 1]] <- weight[[n.layer + 1]] - dw[[n.layer + 1]] -
        l1.reg*((weight[[n.layer + 1]] > 0) - (weight[[n.layer + 1]] < 0)) -
        l2.reg*weight[[n.layer + 1]]
      bias[[n.layer + 1]] <- bias[[n.layer + 1]] - db[[n.layer + 1]]
      if (learning.rate.adaptive == "adagrad") {

        weight.ss[[n.layer + 1]] <- weight.ss[[n.layer + 1]] + weight[[n.layer + 1]]**2
        bias.ss[[n.layer + 1]]   <- bias.ss[[n.layer + 1]]   + bias[[n.layer + 1]]**2
      }
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
        if(learning.rate.adaptive == "momentum") {

          if(i == 1 & k == 1) {

            dw[[j]] <- d_w[[j]] * learning.rate
            db[[j]] <- (t(one_sample_size[[i]]) %*% d_a[[j]]) * learning.rate
          } else {

            dw[[j]] <- d_w[[j]] * learning.rate + rho * last.dw[[j]]
            db[[j]] <- (t(one_sample_size[[i]]) %*% d_a[[j]]) * learning.rate + rho * last.db[[j]]
          }
        } else if (learning.rate.adaptive == "constant") {

          dw[[j]] <- d_w[[j]] * learning.rate
          db[[j]] <- (t(one_sample_size[[i]]) %*% d_a[[j]]) * learning.rate
        } else if (learning.rate.adaptive == "adagrad") {

          dw[[j]] <- d_w[[j]]/sqrt(weight.ss[[j]] + epsilon) * learning.rate
          db[[j]] <- (t(one_sample_size[[i]]) %*% d_a[[j]])/sqrt(bias.ss[[j]] + epsilon) * learning.rate
        } else if (learning.rate.adaptive == "adadelta") {

          bias.grad <- (t(one_sample_size[[i]]) %*% d_a[[j]])
          weight.egs[[j]] <- weight.egs[[j]] * rho + (1-rho) * d_w[[j]]**2
          bias.egs[[j]]   <-   bias.egs[[j]] * rho + (1-rho) * bias.grad**2
          dw[[j]] <- sqrt(weight.es[[j]] + epsilon) / sqrt(weight.egs[[j]] + epsilon) * d_w[[j]]
          db[[j]] <- sqrt(  bias.es[[j]] + epsilon) / sqrt(  bias.egs[[j]] + epsilon) * bias.grad
          weight.es[[j]] <- weight.es[[j]] * rho + (1-rho) * dw[[j]]**2
          bias.es[[j]]   <-   bias.es[[j]] * rho + (1-rho) * db[[j]]**2
        } else if (learning.rate.adaptive == "adam") {

          bias.grad <- (t(one_sample_size[[i]]) %*% d_a[[j]])
          mt.ind <- mt.ind + 1
          mt.w[[j]] <- mt.w[[j]] * beta1 + (1-beta1) * d_w[[j]]
          mt.b[[j]] <- mt.b[[j]] * beta1 + (1-beta1) * bias.grad
          vt.w[[j]] <- vt.w[[j]] * beta2 + (1-beta2) * d_w[[j]]**2
          vt.b[[j]] <- vt.b[[j]] * beta2 + (1-beta2) * bias.grad**2
          dw[[j]] <- learning.rate * mt.w[[j]] / (1-beta1**mt.ind) / (sqrt(vt.w[[j]] / (1-beta2**mt.ind)) + epsilon)
          db[[j]] <- learning.rate * mt.b[[j]] / (1-beta1**mt.ind) / (sqrt(vt.b[[j]] / (1-beta2**mt.ind)) + epsilon)
        }
        # browser()
        weight[[j]] <- weight[[j]] - dw[[j]] - l1.reg*((weight[[j]] > 0) - (weight[[j]] < 0)) - l2.reg*weight[[j]]
        bias[[j]]   <- bias[[j]]   - db[[j]]
        if (learning.rate.adaptive == "adagrad") {

          weight.ss[[j]] <- weight.ss[[j]] + weight[[j]]**2
          bias.ss[[j]]   <- bias.ss[[j]]   + bias[[j]]**2
        }
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
      if(model.type == "classification") {
        pred <- 1/(exp(-pred) + 1)
        loss[k] <- -sum(w.valid * (y.valid * log(pred) + (1-y.valid) * log(1-pred))) / sum(w.valid)
      } else {
        loss[k] <- sum(w.valid * (y.valid - pred)**2) / sum(w.valid)
      }

      if(is.na(loss[k]) | is.null(loss[k]) | is.nan(loss[k]) | is.infinite(loss[k])) {

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








