library(doParallel)
library(microbenchmark)
f1 <- function(j) {foreach(i = 1:j) %do% {sum(tanh(1:i))}}
f2 <- function(j) {foreach(i = 1:j) %dopar% {sum(tanh(1:i))}}

system.time(f1(10000))
system.time(f2(10000))

microbenchmark(f1(100), f2(100))
registerDoParallel(cores = 4)
getDoParWorkers()
registerDoSEQ()
getDoParWorkers()

cl <- makeCluster(20)
registerDoParallel(cl)
system.time(f2(10000))
registerDoSEQ()
stopCluster(cl)

f3 <- function(j, m, n, l) {
  foreach(i = 1:j) %do% {
    matrix(rnorm(m*n), m, n) %*% (matrix(rnorm(n*l), n, l) %*% rnorm(l))
    1
  }
}
f4 <- function(j, m, n, l) {
  foreach(i = 1:j) %dopar% {
    matrix(rnorm(m*n), m, n) %*% (matrix(rnorm(n*l), n, l) %*% rnorm(l))
    1
  }
}
system.time(f3(100, 100, 100, 100))
system.time(f4(100, 100, 100, 100))
microbenchmark(f3(100, 100, 10, 10), f4(100, 100, 10, 10))

f3.r <- f3(100, 100, 100, 100)
f4.r <- f4(100, 100, 100, 100)

x <- rnorm(100000)
x.m <- matrix(x, 100000, 1)
x.mt <- matrix(x, 1, 100000)
microbenchmark(x %*% x, x.mt %*% x.m)

b <- diag(10)
a <- {foreach(i = 1:10) %dopar% b[i, i]}
unlist(a)



