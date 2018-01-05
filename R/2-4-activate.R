sigmoid <- function(x) {1/(exp(-x)+1)}
sigmoid_ <- function(x) {y <- sigmoid(x); y-y**2}
tanh <- function(x) {base::tanh(x)}
tanh_ <- function(x) {y <- tanh(x); 1-y**2}
relu <- function(x) {(abs(x) + x)/2}
relu_ <- function(x) {
  if(!is.null(attr(class(x), "package")) && attr(class(x) ,"package") %in% "gpuR") {
    x <- x[]
    return(vclMatrix((x > 0)*1, type = "double"))
  } else {
    return((x > 0)*1)
  }
}
prelu <- function(x, a = 0.2) {
  m <- (1+a)/2
  n <- (1-a)/2
  (abs(x)*m) + (x*n)
}
prelu_ <- function(x, a = 0.2) {
  b <- 1-a
  if(!is.null(attr(class(x), "package")) && attr(class(x) ,"package") %in% "gpuR") {
    x <- x[]
    return(vclMatrix((x > 0)*b + a))
  } else {
    return((x > 0)*b + a)
  }
}
elu <- function(x, a = 1) {
  if(!is.null(attr(class(x), "package")) && attr(class(x) ,"package") %in% "gpuR") {
    if(a == 1) {
      x <- x[]
      return(vclMatrix((x > 0)*x + (x <= 0)*(exp(x) - 1)))
    } else {
      x <- x[]
      return(vclMatrix((x > 0)*x + (x <= 0)*a*(exp(x) - 1)))
    }
  } else {
    return((x > 0)*x + (x <= 0)*a*(exp(x) - 1))
  }
}
elu_ <- function(x, a = 1) {
  if(!is.null(attr(class(x), "package")) && attr(class(x) ,"package") %in% "gpuR") {
    if(a == 1) {
      x <- x[]
      return(vclMatrix((x > 0) + (x <= 0)*exp(x)))
    } else {
      x <- x[]
      return(vclMatrix((x > 0) + (x <= 0)*a*exp(x)))
    }
  } else {
    return((x > 0) + (x <= 0)*a*exp(x))
  }
}
celu <- function(x, a = 1) {
  if(!is.null(attr(class(x), "package")) && attr(class(x), "package") %in% "gpuR") {
    x <- x[]
    return(vclMatrix((x > 0)*x + (x <= 0)*a*(exp(x/a) - 1)))
  } else {
    return((x > 0)*x + (x <= 0)*a*(exp(x/a) - 1))
  }
}
celu_ <- function(x, a = 1) {
  if(!is.null(attr(class(x), "package")) && attr(class(x) ,"package") %in% "gpuR") {
    x <- x[]
    return(vclMatrix((x > 0) + (x <= 0)*exp(x/a)))
  } else {
    return((x > 0) + (x <= 0)*exp(x/a))
  }
}
