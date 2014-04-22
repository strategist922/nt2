//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/include/functions/simd/slide.hpp>
#include <boost/simd/memory/allocator.hpp>
#include <boost/fusion/include/at.hpp>
#include <vector>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/arithmetic.hpp>
#include <nt2/sdk/bench/setup/combination.hpp>
#include <nt2/sdk/bench/stats/min.hpp>

#include "../erosion.hpp"
#include "../binary_erosion.hpp"
#include "../apply_stencil.hpp"

using boost::simd::pack;
using namespace nt2::bench;
using namespace nt2;

template<typename Operation, typename T> struct filter
{
  typedef pack<T> type;
  static const std::size_t hh = Operation::height/2;
  static const std::size_t hw = Operation::width/2;

  template<typename Setup>
  filter( Setup const& s )
        :  h_(boost::fusion::at_c<0>(s)+2*hh)
        ,  w_(boost::fusion::at_c<1>(s)+2*type::static_size*hw)
  {
    din.resize(h_*w_);
    dout.resize(h_*w_);

    for(std::size_t i = 0; i<h_*w_; ++i) din[i] = T(i+1);
  }

  friend std::ostream& operator<<(std::ostream& os, filter<Operation,T> const& p)
  {
    return os << "(" << p.h_-2*hh << " x " << p.w_-2*hw*type::static_size << ")";
  }

  void operator()()
  {
    nt2::apply_stencil<type::static_size>( Operation(), din, dout, h_, w_ );
  }

  std::size_t size() const { return (h_-2*hh)*(w_-2*hw*type::static_size); }

  protected:

  std::size_t h_,w_,size_;
  std::vector<T, boost::simd::allocator<T> >  din,dout;
};

NT2_REGISTER_BENCHMARK_TPL( filter
                          , ((naive_erosion_<3,3>))
                            ((naive_erosion_<5,5>))
                            ((naive_erosion_<7,7>))
                            ((erosion_<3,3>))
                            ((erosion_<5,5>))
                            ((erosion_<7,7>))
                            ((naive_binary_erosion_<3,3>))
                            ((naive_binary_erosion_<5,5>))
                            ((naive_binary_erosion_<7,7>))
                            ((binary_erosion_<3,3>))
                            ((binary_erosion_<5,5>))
                            ((binary_erosion_<7,7>))
                          )
{
  std::size_t hmin  = args("hmin",   32);
  std::size_t hmax  = args("hmax", 2048);
  std::size_t hstep = args("hstep",  32);

  std::size_t wmin  = args("wmin",   32);
  std::size_t wmax  = args("wmax", 2048);
  std::size_t wstep = args("wstep",  32);

  run_during_with< filter<T,unsigned char> >( 1.
                                          , and_( arithmetic(hmin,hmax,hstep)
                                                , arithmetic(wmin,wmax,wstep)
                                                )
                                          , cycles_per_element<stats::min_>()
                                          );
}
