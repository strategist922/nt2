//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/include/functions/fastnormcdf.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T> struct sum_scalar
{
  sum_scalar(std::size_t n)
  :  size_(n)
  {
    A.resize(size_*size_);
    B.resize(size_*size_);
    C.resize(size_*size_);
    D.resize(size_*size_);
    tmp.resize(size_);

    for(std::size_t i = 0; i <size_*size_; ++i)
      A[i] = B[i] = C[i] = D[i] = T(i+1);

    for(std::size_t i = 0; i <size_; ++i)
      tmp[i] = T(0);
  }

  void operator()()
  {
    for (std::size_t jj=0, offset=0; jj<size_; jj++, offset+=size_)
    {
      T reduced_value = T(0);

      for (std::size_t ii=0, k=offset; ii<size_; ii++, k++)
      {
        reduced_value += C[k]+D[k];
      }

      tmp[jj] = reduced_value;
    }

    for (std::size_t jj=0, offset=0; jj<size_; jj++, offset+=size_)
    {
      for (std::size_t ii=0, k=offset; ii<size_; ii++, k++)
      {
        A[k] = B[k]/tmp[jj];
      }
    }
  }

  friend std::ostream& operator<<(std::ostream& os, sum_scalar<T> const& p)
  {
    return os << "(" << p.size()<< ")";
  }

  std::size_t size() const { return size_*size_; }

  private:
  std::vector<T> A,B,C,D,tmp;
  std::size_t size_;
};

NT2_REGISTER_BENCHMARK_TPL( sum_scalar, (float) )
{
  std::size_t size_min  = args("size_min" , 8000);
  std::size_t size_max  = args("size_max" , 32000);
  std::size_t size_step = args("size_step",    2);

 run_during_with< sum_scalar<float> > ( 10.
                                      , geometric(size_min,size_max,size_step)
                                      , cycles_per_element<stats::median_>()
                                      );
}
