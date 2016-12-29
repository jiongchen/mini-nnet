#include <iostream>
#include <nnet/nnet.h>
#include <nnet/grad_check.h>

using namespace std;
using namespace nnet;

class classification_network : public nnet::neural_network
{
public:
  size_t configure_net(const size_t dimI, const size_t dimO) {
    // cross-entropy loss function
    loss_ = NNET_LOSS_CE;
    
    layers_.resize(4);
    size_t curr_layer;

    curr_layer = 0;
    layers_[curr_layer] = std::make_shared<input_layer>(dimI+1);

    curr_layer = 1;
    layers_[curr_layer] = std::make_shared<hidden_layer>(9+1, &NNET_SIGMOID);

    curr_layer = 2;
    layers_[curr_layer] = std::make_shared<hidden_layer>(9+1, &NNET_SIGMOID);

    curr_layer = 3;
    layers_[curr_layer] = std::make_shared<output_layer_softmax>(dimO);
    
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
  nnet::Mat in =  nnet::Mat::Random(2, sample_num);
  nnet::Mat out = nnet::Mat::Zero(4, sample_num);
  for (size_t i = 0; i < in.cols(); ++i) {
    if ( in(0, i) > 0 && in(1, i) > 0 )
      out(0, i) = 1;
    if ( in(0, i) < 0 && in(1, i) > 0 )
      out(1, i) = 1;
    if ( in(0, i) < 0 && in(1, i) < 0 )
      out(2, i) = 1;
    if ( in(0, i) > 0 && in(1, i) < 0 )
      out(3, i) = 1;
  }

  shared_ptr<nnet::neural_network> net = make_shared<classification_network>();
  net->configure_net(in.rows(), out.rows());

  nnet::Vec w = nnet::Vec::Random(net->get_weight_num());

#if 1
  g_loss = std::make_shared<nnet::nnet_loss_func>(net, in, out);
  int check_gra = numeric_grad_check(evaluate, 1, w.size(), w.data());
  CHECK(check_gra == 0);
#endif

  // train
  nnet::nnet_train(net, in, out, w);

  // test
  nnet::Vec test_x = nnet::Vec::Random(2);  
  nnet::Vec test_y;
  nnet::nnet_predict(net, w, test_x, test_y);  
  cout << "# test input : " << test_x.transpose() << endl;
  cout << "# predict: " << endl << test_y << endl;

  cout << endl;

  test_x << 0, 3;
  nnet_predict(net, w, test_x, test_y);
  cout << "# test input: " << test_x.transpose() << endl;
  cout << "# predict: " << endl << test_y << endl;

  cout << "[INFO] done" << endl;
  return 0;
}
