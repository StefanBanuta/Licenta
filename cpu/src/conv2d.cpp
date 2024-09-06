#include <iostream>
// self
#include "architectures.h"


using namespace architectures;

Conv2D::Conv2D(std::string _name, const int _in_channels, const int _out_channels, const int _kernel_size, const int _stride)
        : Layer(_name), bias(_out_channels), in_channels(_in_channels), out_channels(_out_channels), kernel_size(_kernel_size), stride(_stride),
          params_for_one_kernel(_in_channels * _kernel_size * _kernel_size),
          offset(_kernel_size * _kernel_size) {
    // varificare parametri
    assert(_kernel_size & 1 && _kernel_size >= 3 && "Dimensiunea trebuie sa fie un numar impar pozitiv !");
    assert(_in_channels > 0 && _out_channels > 0 && _stride > 0);//numar de canale de intrare/iesire mai mare decat 0 iar stride tot asa
    // alocare spatiu
    this->weights.reserve(out_channels);
    for(int o = 0;o < out_channels; ++o) {
        weights.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_" + std::to_string(o)));
    }
    // initializare aleatorie weights si bias
    this->seed.seed(212);
    std::normal_distribution<float> engine(0.0, 1.0);
    for(int o = 0;o < out_channels; ++o) bias[o] = engine(this->seed) / random_times;
    for(int o = 0;o < out_channels; ++o) {
        data_type* data_ptr = this->weights[o]->data;
        for(int i = 0;i < params_for_one_kernel; ++i)
            data_ptr[i] = engine(this->seed) / random_times;
    }
}

// functia forward
std::vector<tensor> Conv2D::forward(const std::vector<tensor>& input) {
    // obtine informatii
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;
    const int length = H * W; //nr total de pixeli
    // dimensiuni pentru stratul de iesire 
    const int out_H = std::floor((H - kernel_size - 2 * padding) / stride) + 1;
    const int out_W = std::floor((W - kernel_size - 2 * padding) / stride) + 1;
    const int out_length = out_H * out_W; // numar total pixeli canal iesire 
    
    const int radius = int((kernel_size - 1) / 2);// distanta pata de mijloc
    // alocare a memoriei pentru prima data
    if(this->output.empty()) {
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)  
            this->output.emplace_back(new Tensor3D(out_channels, out_H, out_W, this->name + "_output_" + std::to_string(b)));
        
        int pos = 0;
        for(int x = -radius;x <= radius; ++x)
            for(int y = -radius; y <= radius; ++y) {
                this->offset[pos] = x * W + y;
                ++pos;
            }
    }
    // propagare inapoi
    if(!no_grad) this->__input = input;
    const int H_radius = H - radius; 
    const int W_radius = W - radius;
    const int window_range = kernel_size * kernel_size; // marimea ferestrei
    const int* const __offset = this->offset.data(); // obtine offset
    // fiecare imagine este convoluta separat
    for(int b = 0;b < batch_size; ++b) {
        data_type* const cur_image_features = input[b]->data;//pointer catre date la imagine curenta
        for(int o = 0;o < out_channels; ++o) { 
            data_type* const out_ptr = this->output[b]->data + o * out_length;// harta de caracteisti pentru iesire
            data_type* const cur_w_ptr = this->weights[o]->data;  
            int cnt = 0; // contor pentru loactia convolutiei
            for(int x = radius; x < H_radius; x += stride) {
                for(int y = radius; y < W_radius; y += stride) { //stride pasul cu care se deplaseaza kernel
                    data_type sum_value = 0.f;
                    const int coord = x * W + y; // localizare pixel
                    for(int i = 0;i < in_channels; ++i) { // parcurgere pe canale
                        const int start = i * length + coord; 
                        const int start_w = i * window_range; 
                        for(int k = 0;k < window_range; ++k)
                            sum_value += cur_image_features[start + __offset[k]] * cur_w_ptr[start_w + k];
                    }
                    sum_value += this->bias[o]; // adaugare bias specific
                    out_ptr[cnt] = sum_value;   //rezultata plasat la pozitia cnt de ieire
                    ++cnt;  
                }
            } 
        }
    }
    return this->output;  
}

// folosire memorei heap pentru optimizare
std::vector<tensor> Conv2D::backward(std::vector<tensor>& delta) {
    // informatii anterioare depre gradient
    const int batch_size = delta.size();
    const int out_H = delta[0]->H;//harta de caracteristici iesire
    const int out_W = delta[0]->W;//harta de caracteristici iesire
    const int out_length = out_H * out_W;//numarul total de elemente
    const int H = this->__input[0]->H;//harta de caracteristici intrare
    const int W = this->__input[0]->W;//harta de caracteristici intarare
    const int length = H * W;
    // prima alocare de spatiu 
    if(this->weights_gradients.empty()) {
        // weights
        this->weights_gradients.reserve(out_channels);
        for(int o = 0;o < out_channels; ++o)
            this->weights_gradients.emplace_back(new Tensor3D(in_channels, kernel_size, kernel_size, this->name + "_weights_gradients_" + std::to_string(o)));
        // bias
        this->bias_gradients.assign(out_channels, 0);
    }
    // resetarea gradientilor
    for(int o = 0;o < out_channels; ++o) this->weights_gradients[o]->set_zero();
    for(int o = 0;o < out_channels; ++o) this->bias_gradients[o] = 0;
  
    for(int b = 0;b < batch_size; ++b) {
        for(int o = 0;o < out_channels; ++o) {
            // gradientul pentru fiecare canal
            data_type* o_delta = delta[b]->data + o * out_H * out_W;
            
            for(int i = 0;i < in_channels; ++i) {
                // pointerul de inceput al datelor de intrare al canului i pentru imaginea b din batch 
                data_type* in_ptr = __input[b]->data + i * H * W;
                // pointer pentru gradinet
                data_type* w_ptr = weights_gradients[o]->data + i * kernel_size * kernel_size;
                // parcurgere toti pixelii din fereastra 
                for(int k_x = 0; k_x < kernel_size; ++k_x) {
                    for(int k_y = 0;k_y < kernel_size; ++k_y) {
                        // pargurgere harta de caracteristici
                        data_type sum_value = 0;
                        for(int x = 0;x < out_H; ++x) {
                            // gradientul de eroare
                            data_type* delta_ptr = o_delta + x * out_W;//pointer de intrare
                            data_type* input_ptr = in_ptr + (x * stride + k_x) * W;//se deplaseaza cat stride nu pixel cu pixel
                            for(int y = 0;y < out_W; ++y) {
                                //calculeaza produsul dintre gradientul de eroare pentru poziția din harta de ieșire și valoarea corespunzătoare din imaginea de intrare.
                                sum_value += delta_ptr[y] * input_ptr[y * stride + k_y];
                            }
                        }
                        // actualizare gradient de greuate
                        w_ptr[k_x * kernel_size + k_y] += sum_value / batch_size;
                    }
                }
            }
            // gradient b
            data_type sum_value = 0;
            // gradientul de iesire la mai multe canale
            for(int d = 0;d < out_length; ++d) sum_value += o_delta[d];
            // imparirea la batch
            bias_gradients[o] += sum_value / batch_size;
        }
    }
    // delta_output gradientul de iesire ed la intrare
    // alocare memorie
    if(this->delta_output.empty()) {
        this->delta_output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->delta_output.emplace_back(new Tensor3D(in_channels, H, W, this->name + "_delta_" + std::to_string(b)));
    }
    
    for(int o = 0;o < batch_size; ++o) this->delta_output[o]->set_zero();
    const int radius = (kernel_size - 1) / 2;
    const int H_radius = H - radius;
    const int W_radius = W - radius;
    const int window_range = kernel_size * kernel_size;
    
    for(int b = 0;b < batch_size; ++b) {
        data_type* const cur_image_features = this->delta_output[b]->data;
        for(int o = 0;o < out_channels; ++o) { // fiecare nucleu de convolutie
            data_type* const out_ptr = delta[b]->data + o * out_length;
            data_type* const cur_w_ptr = this->weights[o]->data;  
            int cnt = 0; // contor pentru inregistrarea locatiei unde este stocat fiecare rezultat convolutiei
            for(int x = radius; x < H_radius; x += stride) {
                for(int y = radius; y < W_radius; y += stride) { // parcurgere imagine cu pas de stride 
                    const int coord = x * W + y; // coordonatelui punctului 
                    for(int i = 0;i < in_channels; ++i) { // pentru fiecare canal
                        const int start = i * length + coord; // offset ul imaginii de inatrare
                        const int start_w = i * window_range; // offset ul greutatii
                        for(int k = 0;k < window_range; ++k) { 
                            cur_image_features[start + offset[k]] += cur_w_ptr[start_w + k] * out_ptr[cnt];
                        }
                    }
                    ++cnt; 
                }
            }
        }
    }
    return this->delta_output;
}

// actualizare
void Conv2D::update_gradients(const data_type learning_rate) {
    assert(!this->weights_gradients.empty());
    // actualizre gradienti
    for(int o = 0;o < out_channels; ++o) {
        data_type* w_ptr = weights[o]->data;
        data_type* wg_ptr = weights_gradients[o]->data;
        for(int i = 0;i < params_for_one_kernel; ++i)
            w_ptr[i] -= learning_rate *  wg_ptr[i];
        bias[o] -= learning_rate *  bias_gradients[o];
    }
}

// salvare weights
void Conv2D::save_weights(std::ofstream& writer) const {
    const int filter_size = sizeof(data_type) * params_for_one_kernel;
    for(int o = 0;o < out_channels; ++o)
        writer.write(reinterpret_cast<const char *>(&weights[o]->data[0]), static_cast<std::streamsize>(filter_size));
    writer.write(reinterpret_cast<const char *>(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}

//incarecare weights
void Conv2D::load_weights(std::ifstream& reader) {
    const int filter_size = sizeof(data_type) * params_for_one_kernel;
    for(int o = 0;o < out_channels; ++o)
        reader.read((char*)(&weights[o]->data[0]), static_cast<std::streamsize>(filter_size));
    reader.read((char*)(&bias[0]), static_cast<std::streamsize>(sizeof(data_type) * out_channels));
}


int Conv2D::get_params_num() const {
    return (this->params_for_one_kernel + 1) * this->out_channels;
}
