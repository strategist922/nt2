//==============================================================================
//         Copyright 2015 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_GENERIC_IFFT_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_GENERIC_IFFT_HPP_INCLUDED

#include <nt2/signal/functions/ifft.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>
#include <nt2/core/utility/assign_swap.hpp>
#include <fftw3.h>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( run_assign_, tag::cpu_
                            , (A0)(A1)(N)
                            , ((ast_<A0, nt2::container::domain>))
                              ((node_ < A1, nt2::tag::ifft_
                                      , N , nt2::container::domain
                                      >
                              ))
                            )

  {
    typedef A0& result_type;
    typedef typename A0::value_type s0_type;
    typedef typename A1::value_type s1_type;
    typedef typename A0::extent_type e0_type;
    typedef typename A1::extent_type e1_type;
    typedef nt2::memory::container<tag::table_, s1_type, e0_type> isemantic;
    typedef nt2::memory::container<tag::table_, s0_type, e1_type> osemantic;

    result_type operator()(A0& o, const A1& i) const
    {
      NT2_AS_TERMINAL_IN (isemantic, in , boost::proto::child_c<0>(i));
      NT2_AS_TERMINAL_OUT(osemantic, out, o);

      nt2::ifft( boost::proto::value(in), boost::proto::value(o) );

      assign_swap(o, out);

      return o;
    }
  };
} }

#endif
