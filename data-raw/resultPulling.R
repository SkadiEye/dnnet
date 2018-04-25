#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  make_option(c("-i", "--in"), type="character", default="out",
              help="indir", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="out",
              help="outfile", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


ind = paste(c(rep(0:9, each=10), 10)[-1], rep(c(1:9, 0), 10), sep='')
# ind[1] <- "1"
# setwd("../../../PersonalizedMedicine/CCLE12vs19jobs/")
perf.all <- list()
for(i in 1:length(ind)) {

  perf <- NULL
  try(perf <- read.csv(paste(opt$in, "/", ind[i], ".performance.csv", sep = '')))
  try(perf.all <- append(perf.all, list(perf)))
}
# perf.all[[64]] <- NULL

perf.stat <- array(0, dim = c(nrow(perf), ncol(perf)-3, length(perf.all)))
for(i in 1:length(perf.all)) {

  perf.stat[, , i] <- as.matrix(perf.all[[i]][, 2:3])
}
apply(perf.stat, 1:2, mean)
apply(perf.stat, 1:2, sd)

write.csv(opt$out)


