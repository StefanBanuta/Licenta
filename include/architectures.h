#ifndef CNN_ARCHITECTURES_H
#define CNN_ARCHITECTURES_H


#include <list>
#include <fstream>

#include "pipeline.h"


namespace architectures {
    using namespace pipeline;

    extern data_type random_times;

    extern bool no_grad;


    class WithoutGrad final {
    public:
        explicit WithoutGrad() {
            architectures::no_grad = true;
        }
        ~WithoutGrad() noexcept {
            architectures::no_grad = false;
        }
    };


    class Layer {
    public:
        const std::string name;  
        std::vector<tensor> output; 
    public:
        Layer(std::string& _name) : name(std::move(_name)) {}
        virtual std::vector<tensor> forward(const std::vector<tensor>& input) = 0;
        virtual std::vector<tensor> backward(std::vector<tensor>& delta) = 0;
        virtual void update_gradients(const data_type learning_rate=1e-4) {}
        virtual void save_weights(std::ofstream& writer) const {}
        virtual void load_weights(std::ifstream& reader) {}
        virtual std::vector<tensor> get_output() const { return this->output; }
    };


    class Conv2D : public Layer {
    private:

        std::vector<tensor> weights; 
        std::vector<data_type> bias; 
        const int in_channels;  
        const int out_channels;
        const int kernel_size;  
        const int stride;       
        const int params_for_one_kernel;   
        const int padding = 0;  
        std::default_random_engine seed;  
         std::vector<int> offset; 

        std::vector<tensor> __input; 
 
        std::vector<tensor> delta_output; 
        std::vector<tensor> weights_gradients; 
        std::vector<data_type> bias_gradients; 
    public:
        Conv2D(std::string _name, const int _in_channels=3, const int _out_channels=16, const int _kernel_size=3, const int _stride=2);
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
        int get_params_num() const;
    };


    class MaxPool2D : public Layer {
    private:

        const int kernel_size;
        const int step;
        const int padding; 
     
        std::vector< std::vector<int> > mask; 
        std::vector<tensor> delta_output; 
        std::vector<int> offset;  
    public:
        MaxPool2D(std::string _name, const int _kernel_size=2, const int _step=2)
                : Layer(_name), kernel_size(_kernel_size), step(_step), padding(0),
                  offset(_kernel_size * _kernel_size, 0) {}

        std::vector<tensor> forward(const std::vector<tensor>& input);

        std::vector<tensor> backward(std::vector<tensor>& delta);

    };


    class ReLU : public Layer  {
    public:
        ReLU(std::string _name) : Layer(_name) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };



    class LinearLayer : public Layer {
    private:
 
        const int in_channels;               
        const int out_channels;               
        std::vector<data_type> weights;       
        std::vector<data_type> bias;          

        std::tuple<int, int, int> delta_shape;
        std::vector<tensor> __input;          
 
        std::vector<tensor> delta_output;     
        std::vector<data_type> weights_gradients; 
        std::vector<data_type> bias_gradients;    
    public:
        LinearLayer(std::string _name, const int _in_channels, const int _out_channels);

        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
    };


 

    class BatchNorm2D : public Layer {
    private:

        const int out_channels;
        const data_type eps;
        const data_type momentum;

        std::vector<data_type> gamma;
        std::vector<data_type> beta;

        std::vector<data_type> moving_mean;
        std::vector<data_type> moving_var;

        std::vector<tensor> normed_input;
        std::vector<data_type> buffer_mean;
        std::vector<data_type> buffer_var;

        std::vector<data_type> gamma_gradients;
        std::vector<data_type> beta_gradients;

        tensor norm_gradients;

        std::vector<tensor> __input;
    public:
        BatchNorm2D(std::string _name, const int _out_channels, const data_type _eps=1e-5, const data_type _momentum=0.1);
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
        void update_gradients(const data_type learning_rate=1e-4);
        void save_weights(std::ofstream& writer) const;
        void load_weights(std::ifstream& reader);
    };



    class Dropout : public Layer {
    private:

        data_type p;
        int selected_num;
        std::vector<int> sequence;
        std::default_random_engine drop;

        std::vector<int> mask;
    public:
        Dropout(std::string _name, const data_type _p=0.5): Layer(_name), p(_p), drop(1314) {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };




    class ResNet {
    public:
        bool print_info = false;
    private:
        std::list< std::shared_ptr<Layer> > layers_sequence;
    public:
        ResNet(const int num_classes=4, const bool batch_norm=false);

        std::vector<tensor> forward(const std::vector<tensor>& input);

        void backward(std::vector<tensor>& delta_start);

        void update_gradients(const data_type learning_rate=1e-4);

        void save_weights(const std::filesystem::path& save_path) const;

        void load_weights(const std::filesystem::path& checkpoint_path);

        cv::Mat grad_cam(const std::string& layer_name) const;
    };
}



#endif 
