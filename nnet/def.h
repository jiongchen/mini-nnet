#ifndef MININET_NUMERIC_DEF_H
#define MININET_NUMERIC_DEF_H

#include <Eigen/Sparse>

namespace nnet {

template <typename T>
class Functional
{
public:
  virtual ~Functional() {}
  virtual size_t Nx() const = 0;
  virtual int Val(const T *x, T *val) const = 0;
  virtual int Gra(const T *x, T *gra) const = 0;
  virtual int Hes(const T *x, std::vector<Eigen::Triplet<T>> *hes) const = 0;
};

}

#endif // NUMERIC_DEF_H
