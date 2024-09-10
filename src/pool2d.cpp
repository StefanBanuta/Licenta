#include "architectures.h"


using namespace architectures;

std::vector<tensor> MaxPool2D::forward(const std::vector<tensor>& input) {
    // obtin informatiile de intrare
    const int batch_size = input.size();
    const int H = input[0]->H;
    const int W = input[0]->W;
    const int C = input[0]->C;
    //calculare dimensiuni dupa aplicarea kernel-ului
    const int out_H = std::floor(((H - kernel_size + 2 * padding) / step)) + 1;
    const int out_W = std::floor(((W - kernel_size + 2 * padding) / step)) + 1;
    
    if(this->output.empty()) {//vector gol(prima etapa de pooling)
        // alocare spatiu iesire
        this->output.reserve(batch_size);
        for(int b = 0;b < batch_size; ++b)
            this->output.emplace_back(new Tensor3D(C, out_H, out_W, this->name + "_output_" + std::to_string(b)));//creeaza tensori 3D
        // backpropagation
        if(!no_grad) {//pastram gradientele
            this->delta_output.reserve(batch_size);
            for(int b = 0;b < batch_size; ++b)
                this->delta_output.emplace_back(new Tensor3D(C, H, W, this->name + "_delta_" + std::to_string(b)));
            //mask pentru a pastra pozitiile valorilor maxime dupa pooling 
            this->mask.reserve(batch_size);
            for(int b = 0;b < batch_size; ++b)
                this->mask.emplace_back(std::vector<int>(C * out_H * out_W, 0));
        }
        // calculare offset(diferenta de pozitie)
        int pos = 0;
        for(int i = 0;i < kernel_size; ++i)
            for(int j = 0;j < kernel_size; ++j)
                this->offset[pos++] = i * W + j;
    }
    // completeaza masca cu zero pemtru fiecare pas 
    const int out_length = out_H * out_W;
    int* mask_ptr = nullptr;
    if(!no_grad) {
        const int mask_size = C * out_length;
        for(int b = 0;b < batch_size; ++b) {
            int* const mask_ptr = this->mask[b].data();
            for(int i = 0;i < mask_size; ++i) mask_ptr[i] = 0;
        }
    }
    // gruparea
    const int length = H * W;
    const int H_kernel = H - kernel_size;
    const int W_kernel = W - kernel_size;
    const int window_range = kernel_size * kernel_size;
    for(int b = 0;b < batch_size; ++b) { //fiecare imagine din batch 
        // 16 X 111 X 111 → 16 X 55 X 55
        for(int i = 0;i < C; ++i) {  //fiecare canal
            //pointer la datele canalului de intrare
            data_type* const cur_image_features = input[b]->data + i * length;
            // pointer la locatia de iesire unde se vor scrie rezultatele dupa pooling 
            data_type* const output_ptr = this->output[b]->data + i * out_length;
            // pentru propagare
            if(!no_grad) mask_ptr = this->mask[b].data() + i * out_length;
            int cnt = 0;  //pozitia iesirii din pooling
            for(int x = 0; x <= H_kernel; x += step) {//deplasare kernel
                data_type* const row_ptr = cur_image_features + x * W; //obtinem randul
                for(int y = 0; y <= W_kernel; y += step) {
                    // gasire valoare maxima
                    data_type max_value = row_ptr[y];
                    int max_index = 0; // index pozitia maxima
                    for(int k = 1; k < window_range; ++k) { // se incepe de la 1 pentru ca se face comparatia cu 0 
                        data_type comp = row_ptr[y + offset[k]];
                        if(max_value < comp) {
                            max_value = comp;
                            max_index = offset[k];
                        }
                    }
                    // valorea maxima setata la index pentru back
                    output_ptr[cnt] = max_value;
                    // inregistrare masca pentr mers inapoi
                    if(!no_grad) {
                        max_index += x * W + y; // pozitia valorii maxime gasite in fereastra
                        mask_ptr[cnt] = i * length + max_index;
                    }
                    ++cnt;
                }
            } 
        }
    }
    return this->output;
}

// propagare inversa
std::vector<tensor> MaxPool2D::backward(std::vector<tensor>& delta) {
    // informatii despre gradient
    const int batch_size = delta.size();
    
    for(int b = 0;b < batch_size; ++b) this->delta_output[b]->set_zero();
    
    const int total_length = delta[0]->get_length();
    for(int b = 0;b < batch_size; ++b) {
        int* mask_ptr = this->mask[b].data();
        // adresa de pornire a gradientului 
        data_type* const src_ptr = delta[b]->data;
 
        data_type* const res_ptr = this->delta_output[b]->data;
        for(int i = 0;i < total_length; ++i)
            res_ptr[mask_ptr[i]] = src_ptr[i]; 
    }
    return this->delta_output;
}