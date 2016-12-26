#include <iostream>
#include <nnet/nnet.h>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
  srand(time(NULL));
  const size_t sample_num = 200;

  MatrixXd in =  MatrixXd::Random(3, sample_num);
  MatrixXd out = MatrixXd::Zero(3, sample_num);
  for (size_t i = 0; i < in.cols(); ++i) {
    out(0, i) = sin(in(0, i));
    out(1, i) = cos(in(1, i));
    out(2, i) = exp(in(2, i));
  }

  shared_ptr<nnet::neural_network> net = make_shared<nnet::neural_network>();
  net->configure_net();

  // initialize weights for nnet, close to zero
  VectorXd w = VectorXd::Random(net->get_weight_num());

  nnet::nnet_train(net, in, out, w);

  VectorXd test_x(3);
  test_x << M_PI/2, 0, 1;
  VectorXd test_y;
  nnet::nnet_predict(net, test_x, test_y);
  cout << "# predict: " << test_y.transpose() << endl;

  cout << "[INFO] done" << endl;
  return 0;
}
