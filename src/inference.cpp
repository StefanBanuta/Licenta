// C++
#include <string>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// self
#include "func.h"
#include "architectures.h"

namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

int main() {

    std::setbuf(stdout, 0);

    using namespace architectures;
    std::cout << "inference\n";


    const std::vector<std::string> categories({"Apple___Apple_scab", "Apple___healthy", "Grape___Esca_(Black_Measles)", "Grape___healthy"});


    const int num_classes = categories.size(); 
    ResNet network(num_classes);


    network.load_weights("../checkpoints/ResNet_aug_1e-3/iter_6000_train_0.933_valid_0.959.model");


    std::vector<std::string> images_list({
        "../../datasets/images/Apple___Apple_scab.jpg",
        "../../datasets/images/Apple___healthy.jpg",
        "../../datasets/images/Grape___healthy.jpg",
        "../../datasets/images/Grape___scrab.JPG"
    });


    const std::tuple<int, int, int> image_size({3, 224, 224});
    tensor buffer_data(new Tensor3D(image_size, "inference_buffer"));
    std::vector<tensor> image_buffer({buffer_data});


    WithoutGrad guard;


    for(const auto& image_path : images_list) {

        cv::Mat origin = cv::imread(image_path);
        if(origin.empty() || !std::filesystem::exists(image_path)) {
            std::cout << "Failed to read image file  " << image_path << "\n";
            continue;
        }

        cv::resize(origin, origin, {std::get<1>(image_size), std::get<2>(image_size)});

        image_buffer[0]->read_from_opencv_mat(origin.data);

        const auto output = network.forward(image_buffer);

        const auto prob = softmax(output);

        const int max_index = prob[0]->argmax();
        std::cout << image_path << "===> [classification: " << categories[max_index] << "] [prob: " << prob[0]->data[max_index] << "]\n";
        cv_show(origin);
    }
}