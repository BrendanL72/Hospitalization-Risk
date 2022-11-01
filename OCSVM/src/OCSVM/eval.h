#ifndef _EVAL_H
#define _EVAL_H

#include <stdio.h>
#include <vector>
#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* cross validation function */
double binary_class_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold);

/* predict function */
void binary_class_predict(FILE *input, FILE *output); 

typedef std::vector<double> dvec_t;
typedef std::vector<int>    ivec_t;

// prototypes of evaluation functions
double precision(const dvec_t& dec_values, const ivec_t& ty);
double recall(const dvec_t& dec_values, const ivec_t& ty);
double fscore(const dvec_t& dec_values, const ivec_t& ty);
double bac(const dvec_t& dec_values, const ivec_t& ty);
double auc(const dvec_t& dec_values, const ivec_t& ty);
double accuracy(const dvec_t& dec_values, const ivec_t& ty);
double ap(const dvec_t& dec_values, const ivec_t& ty);

extern struct svm_model* model;
void exit_input_error(int line_num);

#ifdef __cplusplus
}
#endif


#endif
