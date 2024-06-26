//C++
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>
// self
#include "func.h"
#include "metrics.h"
#include "architectures.h"



int main() {

    std::setbuf(stdout, 0);

    std::cout << "opencv  :  " << CV_VERSION << std::endl;

    using namespace architectures;

    const int train_batch_size = 4;
    const int valid_batch_size = 1;
    const int test_batch_size = 1;
    assert(train_batch_size >= valid_batch_size && train_batch_size >= test_batch_size); 
    assert(valid_batch_size == 1 && test_batch_size == 1); 
    const std::tuple<int, int, int> image_size({255, 255, 3});
    const std::filesystem::path dataset_path("../../datasets/plants");
    const std::vector<std::string> categories({"Apple___Apple_scab", "Apple___healthy", "Grape___Esca_(Black_Measles)" , "Grape___healthy"});


    auto dataset = pipeline::get_images_for_classification(dataset_path, categories);


    pipeline::DataLoader train_loader(dataset["train"], train_batch_size, false, true, image_size);
    pipeline::DataLoader valid_loader(dataset["valid"], valid_batch_size, false, false, image_size);


    const int num_classes = categories.size(); 
    AlexNet network(num_classes, false);


    const std::filesystem::path checkpoints_dir("../checkpoints/ResNet_1");
    if(!std::filesystem::exists(checkpoints_dir))
        std::filesystem::create_directories(checkpoints_dir);
    std::filesystem::path best_checkpoint;  
    float current_best_accuracy = -1; 


    const int start_iters = 1;        
    const int total_iters = 20000;   
    const float learning_rate = 1e-3; 
    const int valid_inters = 2000;    
    const int save_iters = 5000;      
    float mean_loss = 0.f;            
    float cur_iter = 0;               
    ClassificationEvaluator train_evaluator;  
    std::vector<int> predict(train_batch_size, -1); 

    for(int iter = start_iters; iter <= total_iters; ++iter) {
        
        const auto sample = train_loader.generate_batch();
  
        const auto output = network.forward(sample.first);
        
        const auto probs = softmax(output);
        
        auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
        mean_loss += loss_delta.first;
        
        network.backward(loss_delta.second);
        
        network.update_gradients(learning_rate);
        
        for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); 
        train_evaluator.compute(predict, sample.second);
        
        ++cur_iter;
        printf("\rTrain===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", iter, total_iters, mean_loss / cur_iter, train_evaluator.get());
        
        if(iter % valid_inters == 0) {
            printf("Începerea validării\n");
            WithoutGrad guard;  
            float mean_valid_loss = 0.f;
            ClassificationEvaluator valid_evaluator;  
            const int samples_num = valid_loader.length();  
            for(int s = 1;s <= samples_num; ++s) {
                const auto sample = valid_loader.generate_batch();
                const auto output = network.forward(sample.first);
                const auto probs = softmax(output);
                const auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
                mean_valid_loss += loss_delta.first;
                for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); 
                valid_evaluator.compute(predict, sample.second);
                printf("\rValid===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", s, samples_num, mean_valid_loss / s, valid_evaluator.get());
            }
            printf("\n\n");
            
            if(iter % save_iters == 0) {
                
                const float train_accuracy = train_evaluator.get();
                const float valid_accuracy = valid_evaluator.get();
                
                std::string save_string("iter_" + std::to_string(iter));
                save_string +=  "_train_" + float_to_string(train_accuracy, 3);
                save_string +=  "_valid_" + float_to_string(valid_accuracy, 3) + ".model";
                std::filesystem::path save_path = checkpoints_dir / save_string;
                
                network.save_weights(save_path);
                
                if(valid_accuracy > current_best_accuracy) {
                    best_checkpoint = save_path;
                    current_best_accuracy = valid_accuracy;
                }
            }
            
            cur_iter = 0;
            mean_loss = 0.f;
            train_evaluator.clear();
        }
    }
    std::cout << "训练结束!\n";

    {
        
        network.load_weights(best_checkpoint);
        
        pipeline::DataLoader test_loader(dataset["test"], test_batch_size, false, false, image_size);
        
        WithoutGrad guard;  
        float mean_test_loss = 0.f;
        ClassificationEvaluator test_evaluator;  
        const int samples_num = test_loader.length();  
        for(int s = 1;s <= samples_num; ++s) {
            const auto sample = test_loader.generate_batch();
            const auto output = network.forward(sample.first);
            const auto probs = softmax(output);
            const auto loss_delta = cross_entroy_backward(probs, one_hot(sample.second, num_classes));
            mean_test_loss += loss_delta.first;
            for(int b = 0;b < train_batch_size; ++b) predict[b] = probs[b]->argmax(); 
            test_evaluator.compute(predict, sample.second);
            printf("\rTest===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", s, samples_num, mean_test_loss / s, test_evaluator.get());
        }
    }
    return 0;
}
