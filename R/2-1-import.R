importData <- function(x, y, w = rep(1, length(y))) {

  new("dnnetInput", x=as.matrix(x), y=y, w=w)
}

getSplit <- function(split, n) {

  if(is.numeric(split) && length(split) == 1 && split < 1)
    split <- sample(n, floor(n * split))

  if(is.numeric(split) && length(split) == 1 && split > 1)
    split <- 1:split

  if(is.character(split) && length(split) == 1 && split == "bootstrap")
    split <- sample(n, replace = TRUE)

  split
}

splitData <-function(object, split) {

  split <- getSplit(split, dim(object@x)[1])

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
