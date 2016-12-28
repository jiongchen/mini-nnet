#ifndef MINI_NNET_H
#define MINI_NNET_H

#include <Eigen/Dense>

#include "config.h"
#include "def.h"
#include "optimizer.h"

namespace nnet {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

typedef double (*nnet_act_func)(const double x, const short order);

inline double NNET_CONSTANT(const double x, const short order) {
  switch ( order ) {
    case 0: return 1;
    case 1: return 0;
  }
}

inline double NNET_IDENTITY(const double x, const short order) {
  switch ( order ) {
    case 0: return x;
    case 1: return 1;
  }
}

inline double NNET_SIGMOID (const double x, const short order) {
  switch ( order ) {
    case 0: return 1/(1+exp(-x));
    case 1: return [](const double z)->double { return z/pow(1+z, 2); }(exp(-x));
  }
}

inline double NNET_RELU    (const double x, const short order) {
  switch ( order ) {
    case 0: return std::max(x, 0.0);
    case 1: return x < 0 ? 0 : 1;
  }
}

struct nnet_layer
{
  nnet_layer(const size_t dim) : dim_(dim) {
    x_ = y_ = dydx_ = dEdx_ = dEdy_ = Vec::Zero(dim_);
    activate_.resize(dim_);
  }
  void apply(const short order) {
    switch ( order ) {
      case 0: for (size_t i = 0; i < dim_; ++i) y_[i]    = activate_[i](x_[i], 0); break;
      case 1: for (size_t i = 0; i < dim_; ++i) dydx_[i] = activate_[i](x_[i], 1); break;
    }
  }
  const size_t dim_;
  Vec x_, y_, dEdx_, dEdy_, dydx_;
  std::vector<nnet_act_func> activate_;
};

class neural_network
{
public:
  neural_network() : dimW_(0) {}
  virtual ~neural_network() {}
  virtual size_t configure_net(const size_t dimI, const size_t dimO) {
    // design your own network structure here
    layers_.resize(5);
    size_t curr_layer;
    
    // input layer
    curr_layer = 0;
    layers_[curr_layer] = std::make_shared<nnet_layer>(dimI+1);
    for (size_t i = 0; i < layers_[curr_layer]->dim_-1; ++i)
      layers_[curr_layer]->activate_[i] = &NNET_IDENTITY;
    layers_[curr_layer]->activate_.back() = &NNET_CONSTANT;

    // hidden layer 1
    curr_layer = 1;
    layers_[curr_layer] = std::make_shared<nnet_layer>(10+1);
    for (size_t i = 0; i < layers_[curr_layer]->dim_-1; ++i)
      layers_[curr_layer]->activate_[i] = &NNET_SIGMOID;
    layers_[curr_layer]->activate_.back() = &NNET_CONSTANT;

    // hidden layer 2
    curr_layer = 2;
    layers_[curr_layer] = std::make_shared<nnet_layer>(20+1);
    for (size_t i = 0; i < layers_[curr_layer]->dim_-1; ++i)
      layers_[curr_layer]->activate_[i] = &NNET_SIGMOID;
    layers_[curr_layer]->activate_.back() = &NNET_CONSTANT;

    // hidden layer 3
    curr_layer = 3;
    layers_[curr_layer] = std::make_shared<nnet_layer>(10+1);
    for (size_t i = 0; i < layers_[curr_layer]->dim_-1; ++i)
      layers_[curr_layer]->activate_[i] = &NNET_SIGMOID;
    layers_[curr_layer]->activate_.back() = &NNET_CONSTANT;
    
    // output layer
    curr_layer = 4;
    layers_[curr_layer] = std::make_shared<nnet_layer>(dimO);
    for (size_t i = 0; i < layers_[curr_layer]->dim_; ++i)
      layers_[curr_layer]->activate_[i] = &NNET_IDENTITY;

    // allocate memory for weights
    W_.resize(layers_.size()-1);
    dEdW_.resize(layers_.size()-1);
    size_t cnt = 0;
    for (size_t i = 0; i < W_.size(); ++i) {
      W_[i]    = std::make_shared<Mat>(layers_[i+1]->dim_, layers_[i]->dim_); W_[i]->setZero();
      dEdW_[i] = std::make_shared<Mat>(layers_[i+1]->dim_, layers_[i]->dim_); dEdW_[i]->setZero();
      cnt += W_[i]->size();
    }

    return dimW_ = cnt;
  }
  size_t get_weight_num() const {
    return dimW_;
  }
  void set_weights(const double *w, const size_t dim) {
    CHECK(dim == dimW_);

    size_t cnt = 0;
    for (size_t i = 0; i < W_.size(); ++i) {
      std::copy(w+cnt, w+cnt+W_[i]->size(), W_[i]->data());
      cnt += W_[i]->size();
    }
  }
  int predict(const Vec &input, Vec &out) {
    CHECK(input.size()+1 == layers_.front()->dim_);

    layers_.front()->x_.setZero();
    layers_.front()->x_.head(input.size()) = input;
    layers_.front()->apply(0);
    
    for (size_t i = 1; i < layers_.size(); ++i) {
      layers_[i]->x_ = (*W_[i-1])*layers_[i-1]->y_;
      layers_[i]->apply(0);
    }

    out = layers_.back()->y_;
  }
  void forward(const Vec &input, const Vec &target, double *val) {
    CHECK(target.size() == layers_.back()->dim_);
    
    Vec output;
    this->predict(input, output);

    // dump loss function value
    // @TODO: replace by a loss function interface
    *val = 0.5*(output-target).squaredNorm();
  }
  void backward(const Vec &target, double *jac, const size_t dim) {
    CHECK(target.size() == layers_.back()->dim_);
    CHECK(dim == dimW_);

    // @TODO: replace by a loss function interface
    layers_.back()->dEdy_ = layers_.back()->y_-target;

    layers_.back()->apply(1);
    layers_.back()->dEdx_ = layers_.back()->dEdy_.cwiseProduct(layers_.back()->dydx_);

    for (int i = dEdW_.size()-1; i >= 0; --i) {
      *dEdW_[i] = layers_[i+1]->dEdx_*layers_[i]->y_.transpose();      
      // compute previous layer dE/dy
      layers_[i]->dEdy_ = W_[i]->transpose()*layers_[i+1]->dEdx_;
      // compute previous layer dy/dx
      layers_[i]->apply(1);
      // compute previous layer dE/dx = dE/dy*dy/dx
      layers_[i]->dEdx_ = layers_[i]->dEdy_.cwiseProduct(layers_[i]->dydx_);
    }

    // dump loss function gradient
    size_t cnt = 0;
    for (size_t i = 0; i < dEdW_.size(); ++i) {
      std::copy(dEdW_[i]->data(), dEdW_[i]->data()+dEdW_[i]->size(), jac+cnt);
      cnt += dEdW_[i]->size();
    }
  }
private:
  std::vector<std::shared_ptr<nnet_layer>> layers_;
  std::vector<std::shared_ptr<Mat>> W_, dEdW_;
  size_t dimW_;
};

class nnet_mse_func : public Functional<double>
{
public:
  typedef Eigen::Triplet<double> TPL;
  nnet_mse_func(const std::shared_ptr<neural_network> &net, const Mat &in, const Mat &out)
      : net_(net), in_(in), out_(out), dim_(net->get_weight_num()) {
    CHECK(in_.cols() == out_.cols());
    g_ = Vec::Zero(dim_);
  }
  size_t Nx() const {
    return dim_;
  }
  int Val(const double *w, double *val) const {
    net_->set_weights(w, dim_);

    double loss = 0; Vec tmp_grad = Vec::Zero(dim_);
    const_cast<Vec &>(g_).setZero();
    for (size_t i = 0; i < in_.cols(); ++i) {
      net_->forward(in_.col(i), out_.col(i), &loss);
      *val += loss;
      net_->backward(out_.col(i), tmp_grad.data(), tmp_grad.size());
      const_cast<Vec &>(g_) += tmp_grad;
    }
    
    return 0;
  }
  int Gra(const double *w, double *gra) const {
    Eigen::Map<Vec> G(gra, dim_);
    G += g_;
    return 0;
  }
  int Hes(const double *w, std::vector<TPL> *hes) const {
    return __LINE__;
  }
private:
  const size_t dim_;
  const std::shared_ptr<neural_network> &net_;
  const Mat &in_, &out_;
  Vec g_;
};

int nnet_train(const std::shared_ptr<neural_network> &net, const Mat &in, const Mat &out, Vec &w) {
  std::shared_ptr<Functional<double>> func
      = std::make_shared<nnet_mse_func>(net, in, out);

  double prev_err = 0;
  func->Val(w.data(), &prev_err);
  std::cout << "# prev error: " << prev_err << std::endl;
  
  opt_args args = {0, 1e-8, true};
  lbfgs_solve(w.data(), w.size(), func, args);

  double post_err = 0;
  func->Val(w.data(), &post_err);
  std::cout << "# post error: " << post_err << std::endl;
  
  return 0;
}

int nnet_predict(const std::shared_ptr<neural_network> &net, const Vec &w, const Vec &in, Vec &out) {
  CHECK(w.size() == net->get_weight_num());
  net->set_weights(w.data(), w.size());
  return net->predict(in, out);
}

}

#endif
