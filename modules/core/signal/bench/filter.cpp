//==============================================================================
//         Copyright 2014          LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <iostream>

#include <nt2/include/functions/filter.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/stats/median.hpp>
#include <nt2/sdk/bench/args.hpp>

#include <nt2/include/functions/tie.hpp>

#include <nt2/include/functions/linspace.hpp>
#include <nt2/include/functions/ones.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T>
struct test_filter
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  test_filter ( std::size_t const& s)
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filt),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    nt2::tie(res) = nt2::filter ( filt, 1, data );
  }

  friend std::ostream& operator<<(std::ostream& os, test_filter<T> const& p)
  {
    return os << "(" << p.size() << " @ " << size_filt << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    static const std::size_t data_size = 37000;
    nt2::table<T> data;
    static const std::size_t size_filt = 7;
    nt2::table<T> filt;
    nt2::table<T> res;
};


NT2_REGISTER_BENCHMARK_TPL( test_filter, NT2_SIMD_REAL_TYPES )
{
  run_during< test_filter<T> > ( 1.
                               , cycles_per_element<stats::median_>()
                               );
}
