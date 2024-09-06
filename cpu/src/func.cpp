#include "func.h"



namespace {
    inline data_type __exp(const data_type x) {
        if(x >= 88) return FLT_MAX; // previne overflow deoarece 88 este valoarea maxima a unui float32
        else if(x <= -50) return 0.f;//evita calcule inutile
        return std::exp(x);
    }
}


// functia softmax transforma scrorurile in probabilitati 
std::vector<tensor> softmax(const std::vector<tensor>& input) {
    const int batch_size = input.size();
    const int num_classes = input[0]->get_length();
    std::vector<tensor> output;
    output.reserve(batch_size);
    for(int b = 0;b < batch_size; ++b) {
        tensor probs(new Tensor3D(num_classes)); //tesnor de stocare a probabilitatilor
        //stocheaza valoarea maxima a vectorului de intare pentru a nu aparea depasirea
        const data_type max_value = input[b]->max();
        data_type sum_value = 0;
        for(int i = 0;i < num_classes; ++i) {
            probs->data[i] = __exp(input[b]->data[i] - max_value);//exp(xi)
            sum_value += probs->data[i];//sum(exp(xj))
        }
        // suma probabilitatilor 1
        for(int i = 0;i < num_classes; ++i) probs->data[i] /= sum_value;//exp(xi)/sum(exp(xj)) 
        //pentru valori NaN egal 0 
        for(int i = 0;i < num_classes; ++i) if(std::isnan(probs->data[i])) probs->data[i] = 0.f;
        output.emplace_back(std::move(probs));
    }
    return output;
}

std::vector<tensor> one_hot(const std::vector<int>& labels, const int num_classes) {
    const int batch_size = labels.size();
    std::vector<tensor> one_hot_code;
    one_hot_code.reserve(batch_size);
    for(int b = 0;b < batch_size; ++b) {
        tensor sample(new Tensor3D(num_classes));
        for(int i = 0;i < num_classes; ++i)
            sample->data[i] = 0;
        assert(labels[b] >= 0 && labels[b] < num_classes);
        sample->data[labels[b]] = 1.0;
        one_hot_code.emplace_back(sample);
    }
    return one_hot_code;
}

//care calculeaza pierderea de entropie incrucisata și gradientii necesari pentru backpropagation
std::pair<data_type, std::vector<tensor> > cross_entroy_backward(
        const std::vector<tensor>& probs, const std::vector<tensor>& labels) {
    const int batch_size = labels.size();
    const int num_classes = probs[0]->get_length();
    std::vector<tensor> delta;//gradientul de pierdere
    delta.reserve(batch_size);
    data_type loss_value = 0;//toata pierderea
    for(int b = 0;b < batch_size; ++b) {
        tensor piece(new Tensor3D(num_classes));
        for(int i = 0;i < num_classes; ++i) {
            piece->data[i] = probs[b]->data[i] - labels[b]->data[i];
            loss_value += std::log(probs[b]->data[i]) * labels[b]->data[i];//sum(ym-ln(am)) formula entropie incrucisata
        }
        delta.emplace_back(piece);
    }
    loss_value = loss_value * (-1.0) / batch_size;//-sum(ym-ln(am))/batch //pierderea medie
    return std::make_pair(loss_value, delta);//perechea de medie si gradient
}

// float to string funct 
std::string float_to_string(const float value, const int precision) {
    std::stringstream buffer;
	buffer.precision(precision);
	buffer.setf(std::ios::fixed);
	buffer << value;
	return buffer.str();
}