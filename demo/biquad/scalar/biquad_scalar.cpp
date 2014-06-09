//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <iostream>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;

template<typename T> struct biquad_scalar
{
  biquad_scalar ( std::size_t s )
                : size_(s)
                , a1(1), a2(1), b0(1), b1(2), b2(1)
                , signal(size_), output(size_)
  {}

  /// Adaptation of ITU G729C codec code
  void operator()()
  {
    T xm1 = 0, xm2 = 0, ym1 = 0, ym2 = 0;
    for(std::size_t i = 0; i < size_; ++i)
    {
      T x0 = signal[i];
      T y0 = ym1*a1 + ym2*a2 + x0*b0 + xm1*b1 + xm2*b2;
      output[i] = y0;

      ym2 = ym1; ym1 = y0;
      xm2 = xm1; xm1 = x0;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, biquad_scalar<T> const& p)
  {
    return os << "(" << p.size_ << ")";
  }

  std::size_t size() const { return size_; }

  private:
  std::size_t size_;
  T a1,a2,b0,b1,b2;
  std::vector<T>  signal, output;
};

NT2_REGISTER_BENCHMARK_TPL( biquad_scalar, (float)(double) )
{
  std::size_t lmin  = args("lmin", 256);
  std::size_t lmax  = args("lmax", 4096);
  std::size_t lstep = args("lstep",  2);

  run_during_with< biquad_scalar<T> > ( 1.
                                      , geometric(lmin,lmax,lstep)
                                      , cycles_per_element<stats::median_>()
                                      );
}
