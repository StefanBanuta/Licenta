#ifndef CNN_FUNC_H
#define CNN_FUNC_H

#include "data_format.h"



std::vector<tensor> softmax(const std::vector<tensor>& input);


std::vector<tensor> one_hot(const std::vector<int>& labels, const int num_classes);

std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
        const std::vector<tensor>& probs, const std::vector<tensor>& labels);


std::string float_to_string(const float value, const int precision);

#endif 
