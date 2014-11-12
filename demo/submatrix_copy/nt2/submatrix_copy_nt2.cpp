//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/table.hpp>

#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/mtimes.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/reshape.hpp>

#include <nt2/sdk/meta/as.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/unit/details/prng.hpp>
#include <nt2/sdk/bench/metric/absolute_time.hpp>
#include <nt2/sdk/bench/metric/speedup.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/protocol/until.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include <iostream>

#include "get_copy.hpp"

using namespace nt2;
using namespace nt2::bench;

template<typename T> struct submatrix_copy_nt2
{

  void operator()()
  {
    int max_steps = 10;

    nt2::table<T> * fout = &fcopy;
    nt2::table<T> * fin  = &f;

    for(int step = 0; step<max_steps; step++)
    {
        get_f<T>( *fin, *fout, nx, ny);

        std::swap(fout,fin);
     }

   }

  friend std::ostream& operator<<(std::ostream& os, submatrix_copy_nt2<T> const& p)
  {
    return os << "(" << p.size()<< ")";
  }

  int size() const { return nx*ny; }

  submatrix_copy_nt2(int size_)
  : nx(size_), ny(size_/2)
  {
    f     = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());
    fcopy = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());
  }

  private:

  // Domain, space and time step
   int nx, ny;
   nt2::table<T> f,fcopy;
};

NT2_REGISTER_BENCHMARK_TPL( submatrix_copy_nt2, (float) )
{
  run_until_with< submatrix_copy_nt2<T> > ( 3., 10
                                  , fixed(64)
                                  , absolute_time<stats::median_>()
                                  );
}
