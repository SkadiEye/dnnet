#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
NumericMatrix activ(NumericMatrix X, String fun, double a) {

  NumericMatrix X_(X.nrow(), X.ncol());
  // if(fun == "sigmoid") return(1/(1+exp(-X)));
  // if(fun == "tanh") X_(i, j) = tanh0(X(i, j));
  // if(fun == "relu") X_(i, j) = relu(X(i, j));
  // if(fun == "prelu") X_(i, j) = prelu(X(i, j), a);
  // if(fun == "elu") X_(i, j) = elu(X(i, j), a);
  // if(fun == "celu") X_(i, j) = celu(X(i, j), a);
  if(fun == "sigmoid") {

    for(int i = 0; i < X.nrow(); i++)
      for(int j = 0; j < X.ncol(); j++)
        X_(i, j) = 1/(1+exp(-X(i, j)));
  } else if(fun == "relu") {

    for(int i = 0; i < X.nrow(); i++)
      for(int j = 0; j < X.ncol(); j++)
        X_(i, j) = (X(i, j) > 0)*X_(i, j);
  }

  return(X_);
}

// [[Rcpp::export]]
NumericMatrix activ2(NumericMatrix X, Function fun, double a) {

  return(wrap(fun(X)));
}

// [[Rcpp::export]]
List list_mat() {

  List x(4);
  List z(4);

  vec h = randu<vec>(5);
  for(int i = 0; i < 4; i++) {

    mat y = randu<mat>(10,5);
    x[i] = y;

    z[i] = as<mat>(x[i])*h;
  }

  return(z);
}

// [[Rcpp::export]]
field<double> mat_sum(List x, Function f) {

  field<mat> y(x.size());
  field<double> z(x.size());
  for(int i = 0; i < x.size(); i++) {

    y(i) = as<mat>(x(i));
    z(i) = mean(mean(as<mat>(f(y(i)))));
  }

  return(z);
}

// [[Rcpp::export]]
field<double> mat_vec(List x, Function f) {

  field<mat> y(x.size());
  field<double> z(x.size());
  for(int i = 0; i < x.size(); i++) {

    y(i) = as<mat>(x(i));
    z(i) = mean(y(i).col(0));
  }

  return(z);
}

// [[Rcpp::export]]
vec pred_dnn(NumericMatrix x, List w, List b, Function f) {

  field<mat> w_(w.size());
  field<vec> b_(w.size());
  mat x_ = as<mat>(x);
  mat pred;
  vec y_pred(x_.n_rows);
  vec one = rep(1.0, x_.n_rows);
  for(int i = 0; i < w.size(); i++) {

    w_(i) = as<mat>(w(i));
    b_(i) = as<vec>(b(i));

    if(i < w.size()-1) {

      if(i == 0) pred = as<mat>(f(x_ * w_(i) + one * b_(i).t()));
      else pred = as<mat>(f(pred * w_(i) + one * b_(i).t()));
    }
  }
  y_pred = pred * w_(w.size() - 1) + one * b_(w.size() - 1).t();
  y_pred = 1/(1+exp(-y_pred));

  return(y_pred);
}

// [[Rcpp::export]]
mat rand_mat(int x, int y) {

  return(randu<mat>(x, y) - 0.5);
}

// [[Rcpp::export]]
void leng_size(NumericVector x) {

  Rcout << x.size() << x.length() << "\n";
}

// [[Rcpp::export]]
void mat_v() {

  mat x = randu<mat>(10, 5);
  mat y = randu<mat>(5, 1);
  vec z = randu<vec>(10);
  vec w = x * y;
  mat u = y * z.t();
  Rcout << w;
  Rcout << u;
}

// [[Rcpp::export]]
void prod_x(vec x, int k) {

  if(prod(x.subvec(0, k-1) > x.subvec(x.size()-k, x.size()-1)) > 0) Rcout << "y";
  else Rcout << "n";
}

// [[Rcpp::export]]
void save_load_field() {

  mat x = randu<mat>(10, 10);
  mat y = randu<mat>(10, 10);
  field<mat> z(2);
  z(0) = x;
  z(1) = y;
  field<mat> w = z;
  z(0)(0, 0) = 1;
  Rcout << w(0)(0, 0);
  Rcout << z(0)(0, 0);
  x(0, 0) = 0;
  Rcout << w(0)(0, 0);
  Rcout << z(0)(0, 0);
}

// [[Rcpp::export]]
SEXP return_field() {

  mat x = randu<mat>(2, 3);
  mat y = randu<mat>(3, 2);
  field<mat> z(2);
  z(0) = x;
  z(1) = y;
  List w(2);
  List w1(2);
  List w2(2);
  w1(0) = wrap(z(0));
  w1(1) = wrap(z(1));
  w2(0) = wrap(z(1));
  w2(1) = wrap(z(0));
  w(0) = w1;
  w(1) = w2;
  return(w);
}








