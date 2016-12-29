#include <iostream>
#include <nnet/nnet.h>
#include <nnet/grad_check.h>

using namespace std;
using namespace nnet;

class regression_network : public nnet::neural_network
{
public:
  size_t configure_net(const size_t dimI, const size_t dimO) {
    // mean squared error
    loss_ = NNET_LOSS_MSE;

    layers_.resize(5);
    size_t curr_layer;

    // input layer
    curr_layer = 0;
    layers_[curr_layer] = std::make_shared<input_layer>(dimI+1);

    // hidden layer 1
    curr_layer = 1;
    layers_[curr_layer] = std::make_shared<hidden_layer>(10+1, &NNET_SIGMOID);
 
    // hidden layer 2
    curr_layer = 2;
    layers_[curr_layer] = std::make_shared<hidden_layer>(20+1, &NNET_SIGMOID);
 
    // hidden layer 3
    curr_layer = 3;
    layers_[curr_layer] = std::make_shared<hidden_layer>(10+1, &NNET_SIGMOID);
    
    // output layer
    curr_layer = 4;
    layers_[curr_layer] = std::make_shared<output_layer_identity>(dimO);
    
    return dimW_ = allocate_memory_for_weights();
  }
};

static std::shared_ptr<nnet::Functional<double>> g_loss;

static void evaluate(const int m, const int n, const double *x,
                     double *value, double *jacobian) {
  *value = 0;
  g_loss->Val(x, value);
  std::fill(jacobian, jacobian+n, 0);
  g_loss->Gra(x, jacobian);
}

int main(int argc, char *argv[])
{
  srand(time(NULL));
  const size_t sample_num = 500;

  // construct training data set
  nnet::Mat in =  nnet::Mat::Random(3, sample_num);
  nnet::Mat out = nnet::Mat::Zero(3, sample_num);
  for (size_t i = 0; i < in.cols(); ++i) {
    out(0, i) = cos(sin(in(0, i)));
    out(1, i) = exp(cos(in(1, i)));
    out(2, i) = pow(exp(in(2, i)), 2);
  }

  shared_ptr<nnet::neural_network> net = make_shared<regression_network>();
  net->configure_net(in.rows(), out.rows());

  // initialize weights for nnet,
  // choose random number close to zero
  nnet::Vec w = nnet::Vec::Random(net->get_weight_num());

#if 1
  g_loss = std::make_shared<nnet::nnet_loss_func>(net, in, out);
  int check_gra = numeric_grad_check(evaluate, 1, w.size(), w.data());
  CHECK(check_gra == 0);
#endif

  // train
  nnet::nnet_train(net, in, out, w);

  // test
  nnet::Vec test_x = nnet::Vec::Random(3);
  nnet::Vec true_y(3);
  true_y << cos(sin(test_x(0))), exp(cos(test_x(1))), pow(exp(test_x(2)), 2);
  
  nnet::Vec test_y;
  nnet::nnet_predict(net, w, test_x, test_y);

  cout << "# expect : " << true_y.transpose() << endl;
  cout << "# predict: " << test_y.transpose() << endl;

  cout << "[INFO] done" << endl;
  return 0;
}
