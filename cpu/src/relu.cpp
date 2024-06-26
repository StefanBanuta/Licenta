
#include "architectures.h"


using namespace architectures;


std::vector<tensor>  ReLU::forward(const std::vector<tensor>& input) {

    const int batch_size = input.size();

    if(output.empty()) {

        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_output_" + std::to_string(b)));
    }

    const int total_length = input[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* const src_ptr = input[b]->data;
        data_type* const out_ptr = this->output[b]->data;
        for(int i = 0;i < total_length; ++i)
            out_ptr[i] = src_ptr[i] >= 0 ? src_ptr[i] : 0;
    }
    return this->output;
}

std::vector<tensor> ReLU::backward(std::vector<tensor>& delta) { 

    const int batch_size = delta.size();

    const int total_length = delta[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* src_ptr = delta[b]->data;
        data_type* out_ptr = this->output[b]->data;
        for(int i = 0;i < total_length; ++i)
            src_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i]; 
    }
    for(int b = 0;b < batch_size; ++b) delta[b]->name = this->name + "_delta_" + std::to_string(b);
    return delta;
}
