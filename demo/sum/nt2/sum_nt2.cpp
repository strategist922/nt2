//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/table.hpp>
#include <nt2/include/functions/sum.hpp>
#include <nt2/include/functions/fma.hpp>
#include <nt2/include/functions/fnms.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/divides.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/unary_minus.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/minus.hpp>
#include <nt2/include/constants/half.hpp>
#include <vector>
#include <iostream>
#include <cstdio>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T> struct sum_nt2
{
  sum_nt2(std::size_t n)
                    :  size_(n)
  {
    A.resize(nt2::of_size(size_,size_));
    B.resize(nt2::of_size(size_,size_));
    C.resize(nt2::of_size(size_,size_));
    D.resize(nt2::of_size(size_,size_));

    for(std::size_t j = 1; j <= size_; ++j)
   {
    for(std::size_t i= 1; i <= size_; ++i)
      A(i,j) = B(i,j) = C(i,j) = D(i,j) = T(i+(j-1)*size_);
   }
}

  void operator()()
  {

    nt2::table<T> A = B/nt2::sum( C + D );
    A.synchronize();
  }

  friend std::ostream& operator<<(std::ostream& os, sum_nt2<T> const& p)
  {
    return os << "(" << p.size() << ")";
  }

  std::size_t size() const { return size_*size_; }

  private:
  nt2::table<T> A, B, C, D;
  std::size_t size_;
};

NT2_REGISTER_BENCHMARK_TPL( sum_nt2, (float) )
{
  std::size_t size_min  = args("size_min" , 8000);
  std::size_t size_max  = args("size_max" , 32000);
  std::size_t size_step = args("size_step",    2);

  run_during_with< sum_nt2<float> > ( 10.
                                                , geometric(size_min,size_max,size_step)
                                                , cycles_per_element<stats::median_>()
                                                );
}
