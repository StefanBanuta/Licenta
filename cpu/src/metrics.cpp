#include "metrics.h"
//eva;uarea modelului 

// calcullez predictiile corecte
void ClassificationEvaluator::compute(const std::vector<int>& predict, const std::vector<int>& labels) {
    const int batch_size = labels.size();  // pentru toate labels
    for(int b = 0;b < batch_size; ++b)
        if(predict[b] == labels[b])
            ++this->correct_num;
    this->sample_num += batch_size;
}
//acuratetea returnata in virgula mobila pentru a nu pierde precizia
float ClassificationEvaluator::get() const {
    return this->correct_num * 1.f / this->sample_num;  
}
//resetare pentru o noua evaluare
void ClassificationEvaluator::clear() {
    this->correct_num = this->sample_num = 0;
}