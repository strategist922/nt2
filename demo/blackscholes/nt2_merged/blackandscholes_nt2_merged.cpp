//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/table.hpp>
#include <nt2/include/functions/log.hpp>
#include <nt2/include/functions/exp.hpp>
#include <nt2/include/functions/fastnormcdf.hpp>
#include <nt2/include/functions/fma.hpp>
#include <nt2/include/functions/fnms.hpp>
#include <nt2/include/functions/sqrt.hpp>
#include <nt2/include/functions/sqr.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/divides.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/unary_minus.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/minus.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/include/constants/half.hpp>
#include <vector>
#include <iostream>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T> struct blackandscholes_nt2_merged
{
  blackandscholes_nt2_merged(std::size_t n)
                    :  size_(n)
  {
    Sa.reuse(nt2::of_size(size_));
    Xa.reuse(nt2::of_size(size_));
    Ta.reuse(nt2::of_size(size_));
    ra.reuse(nt2::of_size(size_));
    va.reuse(nt2::of_size(size_));
    da.reuse(nt2::of_size(size_));
    d1.reuse(nt2::of_size(size_));
    d2.reuse(nt2::of_size(size_));
    R.reuse(nt2::of_size(size_));

    for(std::size_t i = 1; i <= size_; ++i)
      Sa(i) = Xa(i) = Ta(i) = ra(i) = va(i) = T(i);
  }

  void operator()()
  {
    nt2::tie(da,d1,d2,R) = nt2::tie ( nt2::sqrt(Ta)
                                    , nt2::log(Sa/Xa) + (nt2::fma(nt2::sqr(va),T(0.5),ra)*Ta)/(va*da)
                                    , nt2::fnms(va,da,d1)
                                    , nt2::fnms(Xa*nt2::exp(-ra*Ta),nt2::fastnormcdf(d2),Sa*nt2::fastnormcdf(d1))
                                    );
  }

  friend std::ostream& operator<<(std::ostream& os, blackandscholes_nt2_merged<T> const& p)
  {
    return os << "(" << p.size() << ")";
  }

  std::size_t size() const { return size_; }

  private:
  nt2::table<T> Sa, Xa, Ta, ra, va, R, da, d1, d2;
  std::size_t size_;
};

NT2_REGISTER_BENCHMARK_TPL( blackandscholes_nt2_merged, (float) )
{
  std::size_t size_min  = args("size_min",   16);
  std::size_t size_max  = args("size_max", 4096);
  std::size_t size_step = args("size_step",   2);

  run_during_with< blackandscholes_nt2_merged<float> > ( 1.
                                                       , geometric(size_min,size_max,size_step)
                                                       , cycles_per_element<stats::median_>()
                                                       );
}
