#include <vector>
#include <random>

#include "architectures.h"

using namespace architectures;

//consctuctorul clasei unui strat linir
LinearLayer::LinearLayer(std::string _name, const int _in_channels, const int _out_channels)
        : Layer(_name), in_channels(_in_channels), out_channels(_out_channels),
          weights(_in_channels * _out_channels, 0),
          bias(_out_channels) {
    // initializarea aleatoare
    std::default_random_engine e(1998);  //genetaror de numere aleatoare cu seed 1998
    std::normal_distribution<float> engine(0.0, 1.0);//distribuite normala de medie 0 si deviatie 1
    for(int i = 0;i < _out_channels; ++i) bias[i] = engine(e) / random_times;
    const int length = _in_channels * _out_channels;
    for(int i = 0;i < length; ++i) weights[i] = engine(e) / random_times;
}

// W*x + b
// W matricea de weights
// x input 
// b vector de bias

std::vector<tensor> LinearLayer::forward(const std::vector<tensor>& input) {
    //obtinem informatia de intare
    const int batch_size = input.size();
    this->delta_shape = input[0]->get_shape();
    // stergere si incepe iar
    std::vector<tensor>().swap(this->output);
    // alocare memorie pentru tensorul de iesire
    for(int b = 0;b < batch_size; ++b)
        this->output.emplace_back(new Tensor3D(out_channels, this->name + "_output_" + std::to_string(b)));
    // salvare input pentru backpropagation
    if(!no_grad) this->__input = input;
    for(int b = 0;b < batch_size; ++b) {
        data_type* src_ptr = input[b]->data; 
        data_type* res_ptr = this->output[b]->data; 
        for(int i = 0;i < out_channels; ++i) {
            data_type sum_value = 0;
            for(int j = 0;j < in_channels; ++j)
                sum_value += src_ptr[j] * this->weights[j * out_channels + i];// W*x 
            res_ptr[i] = sum_value + bias[i];// W*x + b
        }
    }
    return this->output;
}

std::vector<tensor> LinearLayer::backward(std::vector<tensor>& delta) {
    const int batch_size = delta.size();
    // prima alocare
    if(this->weights_gradients.empty()) {
        this->weights_gradients.assign(in_channels * out_channels, 0);
        this->bias_gradients.assign(out_channels, 0);
    }
    // calcaul gradientul greutatii
    for(int i = 0;i < in_channels; ++i) {
        data_type* w_ptr = this->weights_gradients.data() + i * out_channels; //w_ptr este un pointer catre vectorul de greutati    
        for(int j = 0;j < out_channels; ++j) {
            data_type sum_value = 0;
            for(int b = 0;b < batch_size; ++b)
                sum_value += this->__input[b]->data[i] * delta[b]->data[j];
            w_ptr[j] = sum_value / batch_size;
        }
    }
    // gradientul de bias
    for(int i = 0;i < out_channels; ++i) {
        data_type sum_value = 0;
        for(int b = 0;b < batch_size; ++b)
            sum_value += delta[b]->data[i];
        this->bias_gradients[i] = sum_value / batch_size;
    }
    // initializare memorie pentru eroare
    if(this->delta_output.empty()) {
        this->delta_output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->delta_output.emplace_back(new Tensor3D(delta_shape, "linear_delta_" + std::to_string(b)));
    }
    // calculul gradientului pentru eroare
    for(int b = 0;b < batch_size; ++b) {  
        data_type* src_ptr = delta[b]->data;//eroarea 
        data_type* res_ptr = this->delta_output[b]->data;//noua eroare
        for(int i = 0;i < in_channels; ++i) {  
            data_type sum_value = 0;
            data_type* w_ptr = this->weights.data() + i * out_channels;
            for(int j = 0;j < out_channels; ++j)  
                sum_value += src_ptr[j] * w_ptr[j];
            res_ptr[i] = sum_value;
        }
    }
    // gradientul de pe stratul anterior
    return this->delta_output;
}

//actualizarea greutatii si a biasului 
void LinearLayer::update_gradients(const data_type learning_rate) {
    // verifica daca este gol
    assert(!this->weights_gradients.empty());
    // actualizare gradient descendent
    const int total_length = in_channels * out_channels;
    for(int i = 0;i < total_length; ++i) this->weights[i] -= learning_rate *  this->weights_gradients[i];
    for(int i = 0;i < out_channels; ++i) this->bias[i] -= learning_rate *  this->bias_gradients[i];
}

//salvare greutati 
void LinearLayer::save_weights(std::ofstream& writer) const {
    writer.write(reinterpret_cast<const char *>(&weights[0]), static_cast<std::streamsize>(sizeof(data_type) * in_channels * out_channels));//conversie la un pointer de tip char pentru date in binar reutilizabile
    writer.write(reinterpret_cast<const char *>(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}

// incarcare greutati
void LinearLayer::load_weights(std::ifstream& reader) {
    reader.read((char*)(&weights[0]), static_cast<std::streamsize>(sizeof(data_type) * in_channels * out_channels));
    reader.read((char*)(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}