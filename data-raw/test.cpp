#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List backprop(List weight, List bias,
              NumericMatrix x, NumericVector y, NumericVector w,
              NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid,
              Function activate, Function activate_,
              int n_epoch, int n_batch, string model_type,
              double learning_rate, double l1_reg, double l2_reg, int early_stop_det) {

  Function sample = Environment("package::base")["sample"];

  mat x_ = as<mat>(x);
  vec y_ = as<vec>(y);
  vec w_ = as<vec>(w);
  mat x_valid_ = as<mat>(x_valid);
  vec y_valid_ = as<vec>(y_valid);
  vec w_valid_ = as<vec>(w_valid);

  List weight_(weight.size());
  List bias_(bias.size());
  List a(weight.size());
  List h(weight.size());
  List d_a(weight.size());
  List d_h(weight.size());
  List d_w(weight.size());
  unsigned int sample_size = x.nrow();
  int n_layer = weight.size() - 1;
  vec loss(n_epoch);
  vec one_sample_size = rep(1.0, sample_size);
  double best_loss = INFINITY;
  List best_weight(weight.size());
  List best_bias(weight.size());

  for(int i = 0; i < n_layer + 1; i++) {

    if(i < n_layer) {

      weight_[i] = as<mat>(weight[i]);
      bias_[i] = as<vec>(bias[i]);
    } else {

      weight_[i] = as<vec>(weight[i]);
      bias_[i] = as<double>(bias[i]);
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

    for(int i = 0; i < n_round; i++) {

      mat xi_ = x_.rows(i_bgn[i], i_end[i]);
      vec yi_ = y_.subvec(i_bgn[i], i_end[i]);
      vec wi_ = w_.subvec(i_bgn[i], i_end[i]);
      int n_s = xi_.n_rows;

      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {

          a[j] = xi_ * as<mat>(weight_[j]).t() + one_sample_size * as<vec>(bias_[j]).t();
          h[j] = as<mat>(activate(a[j]));
        } else{

          a[j] = as<mat>(h[j-1]) * as<mat>(weight_[j]).t() + one_sample_size * as<vec>(bias_[j]).t();
          h[j] = as<mat>(activate(a[j]));
        }
      }
      vec y_pi = as<mat>(h[n_layer]) * as<vec>(weight_[n_layer]) + as<double>(bias_[n_layer]);
      if(model_type == "classification")
        y_pi = 1 / (1 + exp(-y_pi));

      d_a[n_layer] = -(yi_ - y_pi) % wi_;
      d_w[n_layer] = as<mat>(h[n_layer - 1]).t() * as<vec>(d_a[n_layer]) / n_s;
      weight_[n_layer] = as<mat>(weight_[n_layer]) - learning_rate * as<mat>(d_w[n_layer]) -
        l1_reg * ((as<mat>(weight_[n_layer]) > 0) - (as<mat>(weight_[n_layer]) < 0)) -
        l2_reg * (as<mat>(weight_[n_layer]));
      bias_[n_layer] = as<double>(bias_[n_layer]) - learning_rate * mean(as<vec>(d_a[n_layer]));
      for(int j = n_layer - 1; j >= 0; j--) {

        if(j == n_layer - 1) {
          d_h[j] = as<vec>(d_a[j + 1]) * as<vec>(weight_[j + 1]).t();
        } else {
          d_h[j] = as<mat>(d_a[j + 1]) * as<mat>(weight_[j + 1]);
        }
        d_a[j] = as<mat>(d_h[j]) % as<mat>(activate_(a[j]));

        if(j > 0) {
          d_w[j] = as<mat>(d_a[j]).t() * as<mat>(h[j - 1]) / n_s;
        } else {
          d_w[j] = as<mat>(d_a[j]).t() * xi_ / n_s;
        }
        weight_[j] = as<mat>(weight_[j]) - learning_rate * as<mat>(d_w[j]) -
          l1_reg * ((as<mat>(weight_[j]) > 0) - (as<mat>(weight_[j]) < 0)) -
          l2_reg * (as<mat>(weight_[j]));
        bias_[j] = as<vec>(bias_[j]) - learning_rate * (as<mat>(d_a[j]) * one_sample_size) / n_s;  // column mean
      }
    }

    if(!Rf_isNull(x_valid)) {

      mat pred;
      vec y_pred;
      for(int j = 0; j < n_layer; j++) {

        if(j == 0) {
          pred = as<mat>(activate(x_valid_ * as<mat>(weight_[j]) + one_sample_size * as<vec>(bias_[j])));
        } else {
          pred = as<mat>(activate(pred * as<mat>(weight_[j]) + one_sample_size * as<vec>(bias_[j])));
        }
      }
      y_pred = as<vec>(activate(pred * as<vec>(weight_[n_layer]) + one_sample_size * as<double>(bias_[n_layer])));
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

