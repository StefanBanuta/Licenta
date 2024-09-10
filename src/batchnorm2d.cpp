// C++
#include <string>
#include <iostream>

#include "architectures.h"

using namespace architectures;


namespace {
    inline data_type square(const data_type x) {
        return x * x;
    }
}

//contructorul BatchNorm2D
BatchNorm2D::BatchNorm2D(std::string _name, const int _out_channels, const data_type _eps, const data_type _momentum)
        : Layer(_name), out_channels(_out_channels), eps(_eps), momentum(_momentum),
          gamma(_out_channels, 1.0), beta(_out_channels, 0),
          moving_mean(_out_channels, 0), moving_var(_out_channels, 0),
          buffer_mean(_out_channels, 0), buffer_var(_out_channels, 0) {}


std::vector<tensor> BatchNorm2D::forward(const std::vector<tensor>& input) {
    // obtinem informatia (consideram imagine patratica)
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;

    //alocarea si initializarea tensorilor
    // verific prima data spatiul de alocare pentru forward
    if(this->output.empty()) {
        this->output.reserve(batch_size); //verific realocarii inutile 
        for(int b = 0;b < batch_size; ++b) this->output.emplace_back(new Tensor3D(out_channels, H, W, this->name + "_output_" + std::to_string(b)));//creare obiect fara copiere
        this->normed_input.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b) this->normed_input.emplace_back(new Tensor3D(out_channels, H, W, this->name + "_normed_" + std::to_string(b)));//creare obiect fara copiere
        this->normed_input.reserve(batch_size);
    }
    //mod antrenare si salveza intarile 
    if(!no_grad) this->__input = input;
    // incepe normalizarea
    const int feature_map_length = H * W;  // dimensiunea unei hartii de caracteristici 2D
    const int output_length = batch_size * feature_map_length;  // numarul de iesiri 
    for(int o = 0;o < out_channels; ++o) {
        if(!no_grad) {
            // media
            data_type u = 0;
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length; //primul element al hartii de caracteristici pentru canalul o din imaginea b
                for(int i = 0;i < feature_map_length; ++i)
                    u += src_ptr[i];
            }
            u = u / output_length;

            // variatia
            data_type var = 0;
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i)
                    var += square(src_ptr[i] - u);
            }
            var = var / output_length;
            if(!no_grad) {
                buffer_mean[o] = u;
                buffer_var[o] = var;
            }
            // normalizare
            const data_type var_inv = 1. / std::sqrt(var + eps);
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length;
                data_type* const des_ptr = output[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i) {
                    norm_ptr[i] = (src_ptr[i] - u) * var_inv;
                    des_ptr[i] = gamma[o] * norm_ptr[i] + beta[o];//transformarea afina(scalare)
                }
            }
            // Actualizați media și varianța istorică (aici trebuie să facem distincția între perioadele de antrenament și perioadele de non-antrenament, adică diferența dintre tren și eval!!!!!!)
            moving_mean[o] = (1 - momentum) * moving_mean[o] + momentum * u;
            moving_var[o] = (1 - momentum) * moving_var[o] + momentum * var;
        }
        else {
            // non_antrenare
            const data_type u = moving_mean[o];
            const data_type var_inv = 1. / std::sqrt(moving_var[o] + eps);
            for(int b = 0;b < batch_size; ++b) {
                data_type* const src_ptr = input[b]->data + o * feature_map_length;
                data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length;
                data_type* const des_ptr = output[b]->data + o * feature_map_length;
                for(int i = 0;i < feature_map_length; ++i) {
                    norm_ptr[i] = (src_ptr[i] - u) * var_inv;
                    des_ptr[i] = gamma[o] * norm_ptr[i] + beta[o];
                }
            }
        }
    }
    return this->output;
}


std::vector<tensor> BatchNorm2D::backward(std::vector<tensor>& delta) {
    //obtinerea informatiei
    const int batch_size = delta.size(); //nr de exemple din batch
    const int feature_map_length = delta[0]->H * delta[0]->W; //dimensiunea hartii de caracteristici
    const int output_length = batch_size * feature_map_length; //dimensiunea totala de elemente
    // alocare spatiu gradienti
    if(gamma_gradients.empty()) {
        gamma_gradients.assign(out_channels, 0); // va contine out_channels elemente toate 0 pentru alocare 
        beta_gradients.assign(out_channels, 0);
        norm_gradients = std::shared_ptr<Tensor3D>(new Tensor3D(batch_size, delta[0]->H, delta[0]->W)); // stocheaza pentru datele normalizate
    }
    // initializarea gradientilor
    for(int o = 0;o < out_channels; ++o) gamma_gradients[o] = beta_gradients[o] = 0;
    //din spate in fata
    for(int o = 0;o < out_channels; ++o) {
        //sterge gradientul med, var, norm
        norm_gradients->set_zero(); // B X H X W
        
        for(int b = 0;b < batch_size; ++b) {
            data_type* const delta_ptr = delta[b]->data + o * feature_map_length; //valorile gradientului erorii din stratul urmator
            data_type* const norm_ptr = normed_input[b]->data + o * feature_map_length; //datele normalizate din acest strat.
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length; //gradientele normalizate care vor fi actulaizate
            for(int i = 0;i < feature_map_length; ++i) {
                gamma_gradients[o] += delta_ptr[i] * norm_ptr[i];
                beta_gradients[o] += delta_ptr[i];
                norm_g_ptr[i] += delta_ptr[i] * gamma[o];
            }
        }
        // in continuare este gradientul varianței var. Media depinde de var, asa ca trebuie sa gasim mai întâi gradientul var
        data_type var_gradient = 0;
        const data_type u = buffer_mean[o];
        const data_type var_inv = 1. / std::sqrt(buffer_var[o] + eps);
        const data_type var_inv_3 = var_inv * var_inv * var_inv;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                var_gradient += norm_g_ptr[i] * (src_ptr[i] - u) * (-0.5) * var_inv_3;
        }
        // gasirea mediei
        data_type u_gradient = 0;
        const data_type inv = var_gradient / output_length;
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                u_gradient += norm_g_ptr[i] * (- var_inv) + inv * (-2) * (src_ptr[i] - u);
        }
        // gradientul returnat
        for(int b = 0;b < batch_size; ++b) {
            data_type* const src_ptr = __input[b]->data + o * feature_map_length;
            data_type* const norm_g_ptr = norm_gradients->data + b * feature_map_length;
            data_type* const back_ptr = delta[b]->data + o * feature_map_length;
            for(int i = 0;i < feature_map_length; ++i)
                back_ptr[i] = norm_g_ptr[i] * var_inv + inv * 2 * (src_ptr[i] - u) + u_gradient / output_length;
        }
    }
    return delta;
}


void BatchNorm2D::update_gradients(const data_type learning_rate) {
    for(int o = 0;o < out_channels; ++o) {
        gamma[o] -= learning_rate * gamma_gradients[o];
        beta[o] -= learning_rate * beta_gradients[o];
    }
}

void BatchNorm2D::save_weights(std::ofstream& writer) const {
    const int stream_size = sizeof(data_type) * out_channels;
    writer.write(reinterpret_cast<const char *>(&gamma[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&beta[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    writer.write(reinterpret_cast<const char *>(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}

void BatchNorm2D::load_weights(std::ifstream& reader) {
    const int stream_size = sizeof(data_type) * out_channels;
    reader.read((char*)(&gamma[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&beta[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_mean[0]), static_cast<std::streamsize>(stream_size));
    reader.read((char*)(&moving_var[0]), static_cast<std::streamsize>(stream_size));
}

