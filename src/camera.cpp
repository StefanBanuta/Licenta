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

    const std::vector<std::string> categories({"Apple___Apple_scab", "Apple___healthy", "Grape___Esca_(Black_Measles)" , "Grape___healthy"});

    const int num_classes = categories.size(); 
    ResNet network(num_classes);

    network.load_weights("../checkpoints/ResNet_aug_1e-3/iter_6000_train_0.933_valid_0.959.model");

    // Setăm dimensiunea imaginii la 224x224
    const std::tuple<int, int, int> image_size({3, 224, 224});
    tensor buffer_data(new Tensor3D(image_size, "inference_buffer"));
    std::vector<tensor> image_buffer({buffer_data});

    WithoutGrad guard;

    // Deschide camera
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        std::cerr << "Camera nu a putut fi deschisă\n";
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Cadru gol capturat\n";
            continue;
        }

       
        // Redimensionare imagine la 224x224 pentru rețea
        cv::resize(frame, frame, {std::get<1>(image_size), std::get<2>(image_size)});
        image_buffer[0]->read_from_opencv_mat(frame.data);

        // Clasificare folosind rețeaua neuronală
        const auto output = network.forward(image_buffer);
        const auto prob = softmax(output);

        const int max_index = prob[0]->argmax();
        std::cout << "Clasificare: " << categories[max_index] << " [probabilitate: " << prob[0]->data[max_index] << "]\n";
        cv_show(frame, "Live Inference");

        // Încheierea buclei la apăsarea unei taste
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
