//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/times.hpp>
#include <boost/simd/include/functions/broadcast.hpp>
// #include <boost/simd/include/functions/selinc.hpp>
// #include <boost/simd/include/functions/seldec.hpp>
// #include <boost/simd/include/functions/if_zero_else_one.hpp>
// #include <boost/simd/include/functions/is_less.hpp>
// #include <boost/simd/include/functions/is_greater.hpp>
#include <boost/simd/include/functions/aligned_store.hpp>
#include <boost/simd/memory/allocator.hpp>
#include <iostream>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/stats/median.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T> struct biquad_scalar
{
  typedef boost::simd::pack<T,4> pack_t;

  biquad_scalar ( std::size_t s )
                : size_(s)
                , signal(size_), output(size_)
  {}

  /// Adaptation of ITU G729C codec code
  void operator()()
  {
    using boost::simd::broadcast;
    using boost::simd::aligned_store;

    pack_t xm1, xm2, ym1, ym2;
    pack_t  coeff_xp3( 0,0,0,1 )
          , coeff_xp2( 0,0,1,3 )
          , coeff_xp1( 0,1,3,5 );

    pack_t coeff_x0 ( 1,3,5,8 )
         , coeff_xm2( 2,3,5,8 )
         , coeff_xm1( 1,1,2,3 );

    pack_t  coeff_ym2( 1,2,3,5 )
          , coeff_ym1( 1,1,2,3 );

    for(std::size_t i = 0; i < size_; i+=2*pack_t::static_size)
    {
      pack_t x0(&signal[i]);

      pack_t sxp0 = broadcast<0>(x0);
      pack_t sxp1 = broadcast<1>(x0);
      pack_t sxp2 = broadcast<2>(x0);
      pack_t sxp3 = broadcast<3>(x0);

      pack_t y0;

      y0  = coeff_xp3 * sxp3;
      y0 += coeff_xp2 * sxp2;
      y0 += coeff_xp1 * sxp1;
      y0 += coeff_x0  * sxp0;
      y0 += coeff_xm1 * xm1;
      y0 += coeff_xm2 * xm2;
      y0 += coeff_ym1 * ym1;
      y0 += coeff_ym2 * ym2;

      aligned_store( y0, &output[i] );

      xm2 = sxp2;
      xm1 = sxp3;
      ym2 = broadcast<2>(y0);
      ym1 = broadcast<3>(y0);
    }
  }

  friend std::ostream& operator<<(std::ostream& os, biquad_scalar<T> const& p)
  {
    return os << "(" << p.size_ << ")";
  }

  std::size_t size() const { return size_; }

  private:
  std::size_t size_;
  std::vector<T, boost::simd::allocator<T> >  signal, output;
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
