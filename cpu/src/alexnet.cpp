// C++
#include <iostream>
// self
#include "architectures.h"

using namespace architectures;

AlexNet::AlexNet(const int num_classes, const bool batch_norm)   {
    this->layers_sequence.emplace_back(new Conv2D("conv_layer_1", 3, 16, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_1", 16));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_1"));
    this->layers_sequence.emplace_back(new MaxPool2D("max_pool_1", 2, 2));

    this->layers_sequence.emplace_back(new Conv2D("conv_layer_2", 16, 32, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_2", 32));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_2"));

    this->layers_sequence.emplace_back(new Conv2D("conv_layer_3", 32, 64, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_3", 64));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_3"));

    this->layers_sequence.emplace_back(new Conv2D("conv_layer_4", 64, 128, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_4", 128));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_4"));

    this->layers_sequence.emplace_back(new Conv2D("conv_layer_5", 128, 256, 3));
    if(batch_norm) this->layers_sequence.emplace_back(new BatchNorm2D("bn_layer_5", 256));
    this->layers_sequence.emplace_back(new ReLU("relu_layer_5"));
    this->layers_sequence.emplace_back(new Dropout("dropout_layer_1", 0.4));

    this->layers_sequence.emplace_back(new LinearLayer("linear_1", 3 * 3 * 256, num_classes));
}

std::vector<tensor> AlexNet::forward(const std::vector<tensor>& input) {
    assert(input.size() > 0);
    if (this->print_info) input[0]->print_shape();

    std::vector<tensor> output(input);
    for (const auto& layer : this->layers_sequence) {
        output = layer->forward(output);
        if (this->print_info) output[0]->print_shape();
    }
    return output;
}

void AlexNet::backward(std::vector<tensor>& delta_start) {
    if (this->print_info) delta_start[0]->print_shape();

    for (auto layer = layers_sequence.rbegin(); layer != layers_sequence.rend(); ++layer) {
        delta_start = (*layer)->backward(delta_start);
        if (this->print_info) delta_start[0]->print_shape();
    }
}

void AlexNet::update_gradients(const data_type learning_rate) {
    for (auto& layer : this->layers_sequence)
        layer->update_gradients(learning_rate);
}

void AlexNet::save_weights(const std::filesystem::path& save_path) const {
    std::ofstream writer(save_path.c_str(), std::ios::binary);
    for (const auto& layer : this->layers_sequence)
        layer->save_weights(writer);
    std::cout << "weights have been saved to " << save_path.string() << std::endl;
    writer.close();
}

void AlexNet::load_weights(const std::filesystem::path& checkpoint_path) {
    if (!std::filesystem::exists(checkpoint_path)) {
        std::cout << "Pre-trained weights file " << checkpoint_path << " does not exist!\n";
        return;
    }
    std::ifstream reader(checkpoint_path.c_str(), std::ios::binary);
    for (auto& layer : this->layers_sequence)
        layer->load_weights(reader);
    std::cout << "Loaded weights from " << checkpoint_path.string() << std::endl;
    reader.close();
}
