#ifndef CNN_PIPELINE_H
#define CNN_PIPELINE_H



// C++
#include <map>
#include <random>
#include <filesystem>
// self
#include "data_format.h"


namespace pipeline {

    using list_type = std::vector<std::pair<std::string, int> >;
    
    std::map<std::string, list_type> get_images_for_classification(
            const std::filesystem::path dataset_path,
            const std::vector<std::string> categories={},
            const std::pair<float, float> ratios={0.8, 0.1});

    
    class ImageAugmentor {
    private:
        std::default_random_engine e, l, c, r; 
        std::uniform_real_distribution<float> engine;
        std::uniform_real_distribution<float> crop_engine;
        std::uniform_real_distribution<float> rotate_engine;
        std::uniform_int_distribution<int> minus_engine;
        std::vector<std::pair<std::string, float> > ops;
    public:
        ImageAugmentor(const std::vector<std::pair<std::string, float> >& _ops={{"hflip", 0.5}, {"vflip", 0.2}, {"crop", 0.7}, {"rotate", 0.5}})
            : e(212), l(826), c(320), r(520),
            engine(0.0, 1.0), crop_engine(0.0, 0.25), rotate_engine(15, 75), minus_engine(1, 10),
            ops(std::move(_ops)) {}
        void make_augment(cv::Mat& origin, const bool show=false);
    };

    class DataLoader {
        using batch_type = std::pair< std::vector<tensor>, std::vector<int> >; 
    private:
        list_type images_list; 
        int images_num;        
        const int batch_size;  
        const bool augment;    
        const bool shuffle;    
        const int seed;        
        int iter = -1;        
        std::vector<tensor> buffer; 
        const int H, W, C;     
    public:
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const std::tuple<int, int, int> image_size={224, 224, 3},  const int _seed=212);
        int length() const;
        batch_type generate_batch();
    private:
        std::pair<tensor, int> add_to_buffer(const int batch_index);
        ImageAugmentor augmentor;
    };
}


#endif 
