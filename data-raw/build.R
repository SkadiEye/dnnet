library(devtools)
library(roxygen2)
library(Rcpp)
# Rcpp::compileAttributes()
# Rcpp.package.skeleton("dnnet", attributes = TRUE, example_code = FALSE, cpp_files = "./R/2-3-dnnet.cpp")
sourceCpp("./R/2-3-dnnet.cpp")
sourceCpp("./data-raw/toy.cpp")
sourceCpp("./data-raw/test.cpp")
sourceCpp("./data-raw/test2.cpp")

library(microbenchmark)

####
x <- matrix(rnorm(1000*100), 1000, 100)
microbenchmark(1/(1+exp(-x)),
               activ(x, "sigmoid", 1),
               activ2(x, sigmoid, 1))

microbenchmark((x>0)*x,
               activ(x, "relu", 1),
               activ2(x, relu, 1))

list_mat_R <- function() {

  x <- list()
  z <- list()
  h <- rnorm(5)
  for(i in 1:4) {
    x[[i]] <- matrix(rnorm(10*5), 10, 5)
    z[[i]] <- x[[i]] %*% h
  }
  z
}
list_mat_R()
list_mat()
microbenchmark(list_mat(), list_mat_R())

####
a1 = matrix(rnorm(10*10), 10, 10)
a2 = matrix(rnorm(100*10), 100, 10)
a3 = rnorm(10)
a4 = rnorm(1)
x = list(a1, a2, a3, a4)
mat_sum(x)
mat_sum(list(a1, a2))
mat_sum(list(a1, a2), exp)
mat_sum(list(a1, a2), function(x) sqrt(abs(x)))
mat_vec(list(as.matrix(a3), as.matrix(a4)), function(x) x)

a1 = matrix(rnorm(20*10), 20, 10)
a2 = matrix(rnorm(10*20), 10, 20)
a3 = as.matrix(rnorm(20))
b1 = rnorm(10)
b2 = rnorm(20)
b3 = rnorm(1)
x = matrix(rnorm(1000, 20), 1000, 20)
pred = pred_dnn(x, list(a1, a2, a3), list(b1, b2, b3), function(x) 1/(1+exp(-x)))

f = function(x) 1/(1+exp(-x))
pred2 = f(f(f(x %*% a1 + rep(1, 1000) %*% t(b1)) %*% a2 + rep(1, 1000) %*% t(b2)) %*% a3 + rep(1, 1000) %*% t(b3))

identical(pred[1:100, ], pred2[1:100, ])

matplot(rand_mat(5, 10))

leng_size(c(1, 1.2, 1.3))
rand_mat(5, 6)
mat_v()
prod_x(7:1, 6)
save_load_field()
return_field()

####
devtools::load_all()
sourceCpp("./R/2-3-dnnet.cpp")
n <- 1000
p <- 10
x <- matrix(rnorm(n*p), n, p)
y <- factor(ifelse(runif(n) > 1/(1+exp(-rowSums(x))), "A", "B"), levels = c("A", "B"))
dat <- importData(x, y)
split.dat <- splitData(dat, "bootstrap")

(a <- Sys.time())
dnn.mod <- dnnet(split.dat$train, validate = split.dat$valid,
                 norm.x = TRUE, norm.y = FALSE,
                 activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-2,
                 l1.reg = 10**-4, l2.reg = 0, n.batch = 100, n.epoch = 100,
                 early.stop = TRUE, early.stop.det = 5, plot = TRUE, accel = "none",
                 learning.rate.adaptive = "adadelta")
(b <- Sys.time() - a)
dnn.mod@loss

plot(y, predict(dnn.mod, x)[, 1])

(a <- Sys.time())
dnn.mod <- dnnet(split.dat$train, validate = split.dat$valid,
                 norm.x = TRUE, norm.y = FALSE,
                 activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-1,
                 l1.reg = 10**-4, l2.reg = 0, n.batch = 100, n.epoch = 100,
                 early.stop = TRUE, early.stop.det = 5, plot = TRUE, accel = "rcpp")
(b <- Sys.time() - a)
dnn.mod@loss

source("../../../unknownProj/stress_test_20171018/dnn-regressor.R")
(a <- Sys.time())
dnn.mod2 <- nn.classifier(split.dat$train@y, split.dat$train@x,
                          validate = data.frame(split.dat$valid@y, split.dat$valid@x),
                          N.hidden = c(10, 10), mu = 10**-1, lambda1 = 10**-4, lambda2 = 0,
                          n.batch = 100, n.epoch = 100, g = elu, g.prime = elu_, f = sigmoid, plot = TRUE)
(b <- Sys.time() - a)
dnn.mod2@loss

microbenchmark(dnnet(split.dat$train, validate = split.dat$valid,
                     norm.x = TRUE, norm.y = FALSE,
                     activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-3,
                     l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 10,
                     early.stop = TRUE, early.stop.det = 5, plot = FALSE),
               dnnet(split.dat$train, ## validate = split.dat$valid,
                     norm.x = TRUE, norm.y = FALSE,
                     activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-3,
                     l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 10,
                     early.stop = TRUE, early.stop.det = 5, plot = FALSE),
               nn.classifier(split.dat$train@y, split.dat$train@x,
                             validate = data.frame(split.dat$valid@y, split.dat$valid@x),
                             N.hidden = c(10, 10), mu = 10**-3, lambda1 = 0, lambda2 = 0,
                             n.batch = 100, n.epoch = 10, g = elu, g.prime = elu.prime, f = sigmoid, plot = FALSE))

####
n <- 1000
p <- 10
x <- matrix(rnorm(n*p), n, p)
y <- rowSums(x) + rnorm(n)
dat <- importData(x, y)
split.dat <- splitData(dat, "bootstrap")

(a <- Sys.time())
set.seed(1000)
dnn.mod <- dnnet(split.dat$train, validate = split.dat$valid,
                 norm.x = TRUE, norm.y = TRUE,
                 activate = "elu", n.hidden = c(20, 10), learning.rate = 2*10**-2,
                 l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                 early.stop = TRUE, early.stop.det = 5, plot = TRUE, accel = "none",
                 learning.rate.adaptive = "adadelta")
(b <- Sys.time() - a)
dnn.mod@loss

(a <- Sys.time())
set.seed(1000)
dnn.mod <- dnnet(split.dat$train, validate = split.dat$valid,
                 norm.x = FALSE, norm.y = FALSE,
                 activate = "elu", n.hidden = c(20, 10), learning.rate = 2*10**-2,
                 l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                 early.stop = TRUE, early.stop.det = 5, plot = TRUE, accel = "rcpp")
(b <- Sys.time() - a)
dnn.mod@loss

plot(y, predict(dnn.mod, x))

(a <- Sys.time())
set.seed(1000)
dnn.mod2 <- nn.regresser(split.dat$train@y, split.dat$train@x,
                         validate = cbind(split.dat$valid@y, split.dat$valid@x),
                         N.hidden = c(20, 10), mu = 2*10**-2, lambda1 = 0, lambda2 = 0,
                         n.batch = 100, n.epoch = 100, g = elu, g.prime = elu_, plot = TRUE)
(b <- Sys.time() - a)
dnn.mod2@loss

microbenchmark(dnnet(split.dat$train, validate = split.dat$valid,
                     norm.x = FALSE, norm.y = FALSE,
                     activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-2,
                     l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 10,
                     early.stop = TRUE, early.stop.det = 5, plot = FALSE, accel = "none"),
               dnnet(split.dat$train, validate = split.dat$valid,
                     norm.x = FALSE, norm.y = FALSE,
                     activate = "elu", n.hidden = c(10, 10), learning.rate = 10**-2,
                     l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 10,
                     early.stop = TRUE, early.stop.det = 5, plot = FALSE, accel = "rcpp"),
               nn.regresser(split.dat$train@y, split.dat$train@x,
                            validate = cbind(split.dat$valid@y, split.dat$valid@x),
                            N.hidden = c(10, 10), mu = 10**-2, lambda1 = 0, lambda2 = 0,
                            n.batch = 100, n.epoch = 10, g = elu, g.prime = elu.prime, plot = FALSE))

####
library(gpuR)
library(pryr)
library(devtools)
library(Rcpp)
library(RcppArmadillo)
use_rcpp()
RcppArmadillo.package.skeleton()





