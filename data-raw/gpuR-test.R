library(gpuR)
library(pryr)
library(microbenchmark)
order <- 1024

A <- matrix(rnorm(order**2), order)
B <- matrix(rnorm(order**2), order)
gpuA <- gpuMatrix(A, type = "double")
gpuB <- gpuMatrix(B, type = "double")

C = A %*% B
gpuC = gpuA %*% gpuB

microbenchmark(A %*% B, gpuA %*% gpuB)

vclA <- vclMatrix(A, type = "double")
vclB <- vclMatrix(B, type = "double")

vclC <- vclA %*% vclB
all.equal(C, gpuC[])
all.equal(C, vclC[])

microbenchmark(A %*% B, gpuA %*% gpuB, vclA %*% vclB)

microbenchmark(deepcopy(gpuA), deepcopy(vclA))


ATA <- A %*% t(A)
gpuATA <- gpuA %*% t(gpuA)
vclATA <- vclA %*% t(vclA)

ata.t <- solve(ATA)
gpuata.t <- solve(gpuATA)
vclata.t <- solve(vclATA)

microbenchmark(solve(ATA), solve(gpuATA), solve(vclATA))
microbenchmark(chol(ATA), chol(gpuATA), chol(vclATA))
microbenchmark(vclMatrix(A), gpuMatrix(A))
microbenchmark(vclMatrix(vclA[1024:1, ], type = "double"),
               gpuMatrix(gpuA[1024:1, ], type = "double"))
X <- diag(order)*0
for(i in 1:1024) X[i, 1025-i] = 1
gpuX <- gpuMatrix(X, type = "double")
vclX <- vclMatrix(X, type = "double")


microbenchmark(X <- diag(order)*0)
microbenchmark(for(i in 1:1024) X[i, 1025-i] = 1)
microbenchmark(sapply(1024:1, function(x) {y <- numeric(1024); y[x] = 1; y}))
microbenchmark(gpuMatrix(X, type = "double"), vclMatrix(X, type = "double"))
microbenchmark(gpuA %*% gpuX, vclA %*% vclX)
microbenchmark(block(vclA, as.integer(1), as.integer(100), as.integer(1), as.integer(100)),
               block(gpuA, as.integer(1), as.integer(100), as.integer(1), as.integer(100)))
microbenchmark(sample(1024))
microbenchmark(vclVector(1:order), gpuVector(1:order))
microbenchmark(vclA %*% vclVector(1:1024))

D <- A[, 1:100]
microbenchmark(D[1024:1, ])
microbenchmark(vclMatrix(D, type = "double"))

class(gpuATA[10:1, 10:20])
class(vclATA[10:1, 10:20])
class(gpuATA)
class(gpuATA + 10)
class(gpuATA + A)
class(vclATA + A)
class(A)
microbenchmark(exp(A), exp(vclA), exp(gpuA))
microbenchmark(sigmoid(A), sigmoid(vclA), sigmoid(gpuA))
microbenchmark((1+exp(-A))**-1, (1+exp(-vclA))**-1, (1+exp(-gpuA))**-1)
microbenchmark((1+exp(-A)), (1+exp(-vclA)), (1+exp(-gpuA)))
microbenchmark(A+1, vclA+1, gpuA+1)
microbenchmark(1/A, 1/vclA, 1/gpuA)
microbenchmark(A**-1, vclA**-1, gpuA**-1)
microbenchmark(A*1, vclA*1, gpuA*1)
microbenchmark(A*2, vclA*2, gpuA*2)
microbenchmark(A-as.integer(1), vclA-as.integer(1), gpuA-as.integer(1))
microbenchmark(A-A**2, vclA-vclA**2, gpuA-gpuA**2)
microbenchmark(A*(1-A), vclA*(1-vclA), gpuA*(1-gpuA))
microbenchmark(tanh(A), tanh(vclA), tanh(gpuA))
microbenchmark(abs(A), abs(vclA), abs(gpuA))
microbenchmark(relu(A), (abs(vclA)+vclA)*0.5, (abs(gpuA)+gpuA)*0.5)
microbenchmark((abs(A) + A)*0.5, (abs(vclA)+vclA)*0.5, (abs(gpuA)+gpuA)*0.5)
microbenchmark(sign(vclA))
microbenchmark((A>0)*1, (abs(A) + A)*0.5/A, (abs(vclA)+vclA)*0.5/vclA, (abs(gpuA)+gpuA)*0.5/gpuA)
microbenchmark(vclA + 1, vclA * 2, vclA / 2)
f <- function(x) {y <- abs(x); y <- y + x; y <- y/2; y/x}
microbenchmark(f(vclA), (abs(vclA) + vclA)*0.5/vclA)
microbenchmark(sigmoid(vclA), vclMatrix(sigmoid(vclA[])))
microbenchmark(vclA*1, vclA*2)
microbenchmark(A[])
microbenchmark(vclA[])

b <- vclMatrix(matrix(1:4, 2, 2), type = "double")
b <- vclVector(1:4, type = "double")
b <- gpuVector(1:4, type = "double")
(b[] > as.vector(1:4))
(as.vector(1:4) > b)

