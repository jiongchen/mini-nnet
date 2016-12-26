#include "optimizer.h"

#include <iostream>
#include <lbfgs.h>

#include "def.h"

using namespace std;

namespace nnet {

static shared_ptr<Functional<double>> energy;

static lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g, const int n,
                                const lbfgsfloatval_t step) {
  double f = 0;
  energy->Val(x, &f);
  std::fill(g, g+n, 0);
  energy->Gra(x, g);
  return f;
}

static int progress(void *instance,
                    const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g,
                    const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm,
                    const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step,
                    int n, int k, int ls) {
  if ( k % 1000 == 0 )
    cout << "----- LBFGS: iter (" << k << "), energy ("
         << fx << "), gnorm (" << gnorm << ")" << endl;
  return 0;
}

int lbfgs_solve(double *x, const size_t dim, const pfunc &f, const opt_args &args) {
  if ( dim != f->Nx() ) {
    cerr << "[error] dim not match\n";
    return __LINE__;
  }

  energy = f;
  lbfgsfloatval_t fx;
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.max_iterations = args.max_iter;

  int ret = lbfgs(dim, x, &fx, evaluate, progress, NULL, &param);

  cout << "L-BFGS optimization terminated with status code = " << ret << endl;
  return 0;
}

}
