//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <vector>
#include <iostream>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/unit/details/prng.hpp>
#include <nt2/sdk/bench/metric/absolute_time.hpp>
#include <nt2/sdk/bench/metric/speedup.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/protocol/until.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include "get_copy.hpp"

using namespace nt2;
using namespace nt2::bench;


template<typename T> struct submatrix_copy_scalar
{
  void operator()()
  {
    int max_steps = 10;
    int bi = 128;
    int bj = 1;

    std::vector<T> * fout = &fcopy;
    std::vector<T> * fin  = &f;

    for(int step = 0; step<max_steps; step++)
    {
     for(int j = 0; j<ny; j+=bj)
     {
       for(int i = 0; i<nx; i+=bi)
       {
        int chunk_i =  (i <= nx-bi) ? bi : nx-i;
        int chunk_j =  (j <= ny-bj) ? bj : ny-j;
        int max_i = i+chunk_i;
        int max_j = j+chunk_j;

        for(int j_ = j; j_<max_j; j_++)
          for(int i_ = i; i_<max_i; i_++)
          {
            get_f<T>(*fin, *fout, nx, ny, i_, j_);
          }
        }
      }

      std::swap(fout,fin);
    }
  }

  friend std::ostream& operator<<(std::ostream& os, submatrix_copy_scalar<T> const& p)
  {
    return os << "(" << p.size()<< ")";
  }

  int size() const { return nx*ny; }


  submatrix_copy_scalar(int size_)
  : nx(size_),ny(size_/2)
  , f (9 * nx * ny, 0.)
  , fcopy(9 * nx * ny)
  {}

  private:
    int nx,ny;
    std::vector<T> f,fcopy;

};

NT2_REGISTER_BENCHMARK_TPL( submatrix_copy_scalar, (float) )
{
  run_until_with< submatrix_copy_scalar<T> > ( 3., 10
                                  , fixed(64)
                                  , absolute_time<stats::median_>()
                                  );
}
