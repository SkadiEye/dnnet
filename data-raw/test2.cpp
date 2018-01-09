#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List backprop(NumericVector n_hidden, double w_ini, // List weight, List bias,
              NumericMatrix x, NumericVector y, NumericVector w,
              NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid,
              Function activate, Function activate_,
              int n_epoch, int n_batch, string model_type,
              double learning_rate, double l1_reg, double l2_reg, int early_stop_det) {

  mat x_ = as<mat>(x);
  vec y_ = as<vec>(y);
  vec w_ = as<vec>(w);
  mat x_valid_ = as<mat>(x_valid);
  vec y_valid_ = as<vec>(y_valid);
  vec w_valid_ = as<vec>(w_valid);

  unsigned int sample_size = x.nrow();
  int n_layer = n_hidden.size();
  field<mat> weight_(n_layer + 1);
  field<vec> bias_(n_layer + 1);
  field<mat> a(n_layer + 1);
  field<mat> h(n_layer + 1);
  field<mat> d_a(n_layer + 1);
  field<mat> d_h(n_layer + 1);
  field<mat> d_w(n_layer + 1);
  vec loss(n_epoch);
  double best_loss = INFINITY;
  field<mat> best_weight(n_layer + 1);
  field<vec> best_bias(n_layer + 1);

  for(int i = 0; i < n_layer + 1; i++) {

    // weight_(i) = as<mat>(weight[i]);
    // bias_(i) = as<vec>(bias[i]);
    if(i == 0) {
      weight_(i) = (randu<mat>(x_.n_cols, n_hidden[i]) - 0.5) * w_ini * 2;
      bias_(i) = (randu<vec>(n_hidden[i]) - 0.5) * w_ini;
    } else if(i == n_layer) {
      weight_(i) = (randu<mat>(n_hidden[i-1], 1)) * w_ini * 2;
      bias_(i) = (randu<vec>(1)) * w_ini;
    } else {
      weight_(i) = (randu<mat>(n_hidden[i-1], n_hidden[i])) * w_ini * 2;
      bias_(i) = (randu<vec>(n_hidden[i])) * w_ini;
    }
  }

  int n_round = ceil(sample_size/n_batch);
  uvec i_bgn(n_round);
  uvec i_end(n_round);

  for(int s = 0; s < n_round; s++) {

    i_bgn[s] = s*n_batch;
    i_end[s] = (s+1)*n_batch - 1;
    if(i_end[s] > sample_size - 1) i_end[s] = sample_size - 1;
  }

  for(int k = 0; k < n_epoch; k++) {

    // shuffle
    uvec new_order = as<uvec>(sample(sample_size, sample_size)) - 1;
    x_ = x_.rows(new_order);
    y_ = y_.elem(new_order);
    w_ = w_.elem(new_order);
    //
    // Rcout << y_.size() << w_.size() << x_.n_cols << x_.n_rows << "\n";

    // for(int i = 0; i < n_round; i++) {
    //
    //   mat xi_ = x_.rows(i_bgn[i], i_end[i]);
    //   vec yi_ = y_.subvec(i_bgn[i], i_end[i]);
    //   vec wi_ = w_.subvec(i_bgn[i], i_end[i]);
    //   int n_s = xi_.n_rows;
    //   vec one_sample_size = rep(1.0, n_s);
    //
    //   Rcout << xi_.n_rows << xi_.n_cols << "\n";
    //
    //   for(int j = 0; j < n_layer; j++) {
    //
    //     if(j == 0) {
    //
    //       a(j) = xi_ * weight_(j) + one_sample_size * bias_(j).t();
    //       // Rcout << as<mat>(a[j]).n_cols << as<mat>(a[j]).n_rows << "\n";
    //       h(j) = as<mat>(activate(a(j)));
    //       // Rcout << as<mat>(h[j]).n_cols << as<mat>(h[j]).n_rows << "\n";
    //     } else{
    //
    //       a(j) = h(j-1) * weight_(j) + one_sample_size * bias_(j).t();
    //       h(j) = as<mat>(activate(a(j)));
    //     }
    //
    //     Rcout << a(j).n_cols << a(j).n_rows << "\n";
    //   }
    //   vec y_pi = h(n_layer - 1) * weight_(n_layer) + bias_(n_layer);
    //   if(model_type == "classification")
    //     y_pi = 1 / (1 + exp(-y_pi));
    //
    //   Rcout << y_pi.size() << "\n";
    //
    //   d_a(n_layer) = -(yi_ - y_pi) % wi_;
    //   d_w(n_layer) = h(n_layer - 1).t() * d_a(n_layer) / n_s;
    //   weight_(n_layer) = weight_(n_layer) - learning_rate * d_w(n_layer) -
    //     l1_reg * ((weight_(n_layer) > 0) - (weight_(n_layer) < 0)) -
    //     l2_reg * (weight_(n_layer));
    //   bias_(n_layer) = bias_(n_layer) - learning_rate * mean(mean(d_a(n_layer)));
    //
    //   Rcout << d_a(n_layer).size() << "\n";
    //   for(int j = n_layer - 1; j >= 0; j--) {
    //
    //     Rcout << j << "\n";
    //     // if(j == n_layer - 1) {
    //     //   d_h(j) = d_a(j + 1) * weight_(j + 1).t();
    //     // } else {
    //     //   d_h(j) = d_a(j + 1) * weight_(j + 1);
    //     // }
    //
    //     d_h(j) = d_a(j + 1) * weight_(j + 1).t();
    //     Rcout << a(j).n_cols << a(j).n_rows << "\n";
    //     d_a(j) = d_h(j) % as<mat>(activate_(a(j)));
    //
    //     Rcout << h(j - 1).n_cols << h(j - 1).n_rows << "\n";
    //     if(j > 0) {
    //       d_w(j) = h(j - 1).t() * d_a(j) / n_s;
    //     } else {
    //       d_w(j) = xi_.t() * d_a(j) / n_s;
    //     }
    //     weight_(j) = weight_(j) - learning_rate * d_w(j) -
    //       l1_reg * ((weight_(j) > 0) - (weight_(j) < 0)) -
    //       l2_reg * (weight_(j));
    //     bias_(j) = bias_(j) - learning_rate * (d_a(j).t() * one_sample_size) / n_s;  // column mean
    //   }
    // }
    //
    if(!Rf_isNull(x_valid)) {

      mat pred;
      int n_s = x_valid_.n_rows;
      vec y_pred(n_s);
      vec one_sample_size = rep(1.0, n_s);
      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {
          pred = as<mat>(activate(x_valid_ * weight_(j) + one_sample_size * bias_(j).t()));
        } else {
          pred = as<mat>(activate(pred * weight_(j) + one_sample_size * bias_(j).t()));
        }
      }
      y_pred = as<vec>(activate(pred * weight_(n_layer) + one_sample_size * bias_(n_layer).t()));
      if(model_type == "classification") {

        y_pred = y_pred / (1 + exp(-y_pred));
        loss[k] = -sum(w_valid_ % (y_valid_ % log(y_pred) + (1-y_valid_) % log(1-y_pred))) / sum(w_valid_);
      } else {

        loss[k] = -sum(w_valid_ % pow(y_valid_ - y_pred, 2)) / sum(w_valid_);
      }

      if(loss[k] == NA_REAL) {

        break;
      } else {

        if(loss[k] < best_loss) {

          best_loss = loss[k];
          best_weight = clone(weight_);
          best_bias = clone(bias_);
        }

        if(k > early_stop_det) {
          if(prod(loss.subvec(k-early_stop_det, k) > loss.subvec(k-early_stop_det-1, k-1)) > 0) {
            break;
          }
        }
      }
    }
  }

  if(!Rf_isNull(x_valid)) {

    return(List::create(Rcpp::Named("weight") = best_weight,
                        Rcpp::Named("bias") = best_bias,
                        Rcpp::Named("loss") = loss));
  } else {

    return(List::create(Rcpp::Named("weight") = weight_,
                        Rcpp::Named("bias") = bias_,
                        Rcpp::Named("loss") = loss));
  }
}

