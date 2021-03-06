// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// backprop
SEXP backprop(NumericVector n_hidden, double w_ini, NumericMatrix x, NumericVector y, NumericVector w, bool valid, NumericMatrix x_valid, NumericVector y_valid, NumericVector w_valid, std::string activ, int n_epoch, int n_batch, std::string model_type, double learning_rate, double l1_reg, double l2_reg, int early_stop_det, std::string learning_rate_adaptive, double rho, double epsilon, double beta1, double beta2, std::string loss_f);
RcppExport SEXP _dnnet_backprop(SEXP n_hiddenSEXP, SEXP w_iniSEXP, SEXP xSEXP, SEXP ySEXP, SEXP wSEXP, SEXP validSEXP, SEXP x_validSEXP, SEXP y_validSEXP, SEXP w_validSEXP, SEXP activSEXP, SEXP n_epochSEXP, SEXP n_batchSEXP, SEXP model_typeSEXP, SEXP learning_rateSEXP, SEXP l1_regSEXP, SEXP l2_regSEXP, SEXP early_stop_detSEXP, SEXP learning_rate_adaptiveSEXP, SEXP rhoSEXP, SEXP epsilonSEXP, SEXP beta1SEXP, SEXP beta2SEXP, SEXP loss_fSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type n_hidden(n_hiddenSEXP);
    Rcpp::traits::input_parameter< double >::type w_ini(w_iniSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< bool >::type valid(validSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type x_valid(x_validSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y_valid(y_validSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w_valid(w_validSEXP);
    Rcpp::traits::input_parameter< std::string >::type activ(activSEXP);
    Rcpp::traits::input_parameter< int >::type n_epoch(n_epochSEXP);
    Rcpp::traits::input_parameter< int >::type n_batch(n_batchSEXP);
    Rcpp::traits::input_parameter< std::string >::type model_type(model_typeSEXP);
    Rcpp::traits::input_parameter< double >::type learning_rate(learning_rateSEXP);
    Rcpp::traits::input_parameter< double >::type l1_reg(l1_regSEXP);
    Rcpp::traits::input_parameter< double >::type l2_reg(l2_regSEXP);
    Rcpp::traits::input_parameter< int >::type early_stop_det(early_stop_detSEXP);
    Rcpp::traits::input_parameter< std::string >::type learning_rate_adaptive(learning_rate_adaptiveSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< double >::type beta1(beta1SEXP);
    Rcpp::traits::input_parameter< double >::type beta2(beta2SEXP);
    Rcpp::traits::input_parameter< std::string >::type loss_f(loss_fSEXP);
    rcpp_result_gen = Rcpp::wrap(backprop(n_hidden, w_ini, x, y, w, valid, x_valid, y_valid, w_valid, activ, n_epoch, n_batch, model_type, learning_rate, l1_reg, l2_reg, early_stop_det, learning_rate_adaptive, rho, epsilon, beta1, beta2, loss_f));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_dnnet_backprop", (DL_FUNC) &_dnnet_backprop, 23},
    {NULL, NULL, 0}
};

RcppExport void R_init_dnnet(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
