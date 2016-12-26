#include <iostream>
#include <nnet/nnet.h>

using namespace std;

int main(int argc, char *argv[])
{
  srand(time(NULL));
  const size_t sample_num = 500;

  // construct training data set
  nnet::Mat in =  nnet::Mat::Random(3, sample_num);
  nnet::Mat out = nnet::Mat::Zero(3, sample_num);
  for (size_t i = 0; i < in.cols(); ++i) {
    out(0, i) = sin(in(0, i));
    out(1, i) = cos(in(1, i));
    out(2, i) = exp(in(2, i));
  }

  shared_ptr<nnet::neural_network> net = make_shared<nnet::neural_network>();
  net->configure_net();

  // initialize weights for nnet,
  // choose random number close to zero
  nnet::Vec w = nnet::Vec::Random(net->get_weight_num());

  // train
  nnet::nnet_train(net, in, out, w);

  // test
  nnet::Vec test_x = nnet::Vec::Random(3);
  nnet::Vec true_y(3);
  true_y << sin(test_x(0)), cos(test_x(1)), exp(test_x(2));
  
  nnet::Vec test_y;
  nnet::nnet_predict(net, test_x, test_y);

  cout << "# expect : " << true_y.transpose() << endl;
  cout << "# predict: " << test_y.transpose() << endl;

  cout << "[INFO] done" << endl;
  return 0;
}
