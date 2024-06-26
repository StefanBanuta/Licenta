#include <string>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
// self
#include "func.h"
#include "architectures.h"

namespace {
    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(1); // Modificat pentru a nu bloca execuția
    }
}

int main() {

    std::setbuf(stdout, 0);

    using namespace architectures;
    std::cout << "inference\n";

    const std::vector<std::string> categories({ "Apple___healthy" , "Apple___Apple_scab", "Grape___Esca_(Black_Measles)", "Grape___healthy"});

    const int num_classes = categories.size(); 
    AlexNet network(num_classes);

    network.load_weights("/checkpoints/AlexNet_aug_1e-3/iter_10000_train_0.956_valid_0.966.model");

    const std::tuple<int, int, int> image_size({3, 255, 255});
    tensor buffer_data(new Tensor3D(image_size, "inference_buffer"));
    std::vector<tensor> image_buffer({buffer_data});

    WithoutGrad guard;

    // Deschide camera
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "Camera nu a putut fi deschisă\n";
        return -1;
    }

    const float probability_threshold = 0.7; // Prag de probabilitate

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Cadru gol capturat\n";
            continue;
        }

        cv::resize(frame, frame, {std::get<1>(image_size), std::get<2>(image_size)});
        image_buffer[0]->read_from_opencv_mat(frame.data);

        const auto output = network.forward(image_buffer);
        const auto prob = softmax(output);

        const int max_index = prob[0]->argmax();
        const float max_probability = prob[0]->data[max_index];

        if (max_probability >= probability_threshold) {
            std::cout << "Clasificare: " << categories[max_index] << " [probabilitate: " << max_probability << "]\n";
        }

        cv_show(frame, "Live Inference");

        // Încheierea buclei la apăsarea unei taste
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
