#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>

namespace nnet {

template <typename T>
class Functional;

struct opt_args {
  size_t max_iter;
  double eps;
  bool lineseach;
};

using pfunc=std::shared_ptr<Functional<double>>;

int lbfgs_solve(double *x, const size_t dim, const pfunc &f, const opt_args &args);

}

#endif
