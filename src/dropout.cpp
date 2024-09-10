// self
#include "architectures.h"

using namespace architectures;


std::vector<tensor> Dropout::forward(const std::vector<tensor>& input) {
    const int batch_size = input.size();
    const int out_channels = input[0]->C;
    const int area = input[0]->H * input[0]->W;//dimensiunea
    if(this->sequence.empty()) {
        this->sequence.assign(out_channels, 0);
        for(int o = 0;o < out_channels; ++o) this->sequence[o] = o;
        this->selected_num = int(p * out_channels); // numarul de canale dezactivate
        assert(out_channels > this->selected_num);
        this->mask.assign(out_channels, 0);
        //atribuire iesire
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) this->output.emplace_back(new Tensor3D(out_channels, input[0]->H, input[0]->W));
    }
    //amestecare
    std::shuffle(this->sequence.begin(), this->sequence.end(), this->drop);
    // antrenare
    if(!no_grad) {
        //marcheaza canalele dezactivate
        for(int i = 0;i < out_channels; ++i)
            this->mask[i] = i >= selected_num ? this->sequence[i] : -1;
        const auto copy_size = sizeof(data_type) * area;
        for(int b = 0;b < batch_size; ++b) {
            for(int o = 0;o < out_channels; ++o) { 
                if(o >= this->selected_num)  
                    std::memcpy(this->output[b]->data + o * area, input[b]->data + o * area, copy_size);
                else std::memset(this->output[b]->data + o * area, 0, copy_size);
            }
        }
    }
    else { // verificare
        // înmulțiți rezultatul cu 1 - p
        const int length = input[0]->get_length();
        const data_type prob = 1 - this->p;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = input[b]->data;
            data_type* const des_ptr = this->output[b]->data;
            for(int i = 0;i < length; ++i) des_ptr[i] = src_ptr[i] * prob;
        }
    }
    return this->output;
}

std::vector<tensor> Dropout::backward(std::vector<tensor>& delta) {
    const int batch_size = delta.size();
    const int out_channels = delta[0]->C;
    const int area = delta[0]->H * delta[0]->W;
    for(int b = 0;b < batch_size; ++b)
        for(int o = 0;o < out_channels; ++o)
            if(this->mask[o] == -1) // canal dezactivat se transmite gradientul 0 inapoi
                std::memset(delta[b]->data + o * area, 0, sizeof(data_type) * area);
    return delta;
}