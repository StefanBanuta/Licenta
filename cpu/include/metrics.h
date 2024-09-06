#ifndef CNN_METRICS_H
#define CNN_METRICS_H

#include <vector>


class ClassificationEvaluator {
private:
    int correct_num = 0;  
    int sample_num = 0;   
public:
    ClassificationEvaluator() = default;
    
    void compute(const std::vector<int>& predict, const std::vector<int>& labels);
    
    float get() const;
    
    void clear();
};



#endif 
