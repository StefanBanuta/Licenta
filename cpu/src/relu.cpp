
#include "architectures.h"


using namespace architectures;


std::vector<tensor>  ReLU::forward(const std::vector<tensor>& input) { //functia de forward

    const int batch_size = input.size();

    if(output.empty()) {  //verific daca este gol

        this->output.reserve(batch_size);//aloca spatiu
        for(int b = 0;b < batch_size; ++b)
            this->output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W, this->name + "_output_" + std::to_string(b)));//creaza tensori 3D 
    }

    const int total_length = input[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* const src_ptr = input[b]->data; //pointer care indica input
        data_type* const out_ptr = this->output[b]->data;//pointer care indica output
        for(int i = 0;i < total_length; ++i)
            out_ptr[i] = src_ptr[i] >= 0 ? src_ptr[i] : 0;//pastreaza valorile mai mari de zero restul devin zero
    }
    return this->output;
}

std::vector<tensor> ReLU::backward(std::vector<tensor>& delta) {  //funct pentru backpropagation

    const int batch_size = delta.size();

    const int total_length = delta[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        data_type* src_ptr = delta[b]->data;//pointer care indica input
        data_type* out_ptr = this->output[b]->data;//pointer care indica output
        for(int i = 0;i < total_length; ++i)
            src_ptr[i] = out_ptr[i] <= 0 ? 0 : src_ptr[i]; //pastreaza valorile mai mari de zero restul devin zero
    }
    for(int b = 0;b < batch_size; ++b) delta[b]->name = this->name + "_delta_" + std::to_string(b);
    return delta;
}
