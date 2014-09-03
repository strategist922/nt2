//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_DETAILS_CONV_SIZE_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_CONV_SIZE_HPP_INCLUDED

#include <nt2/signal/options.hpp>

namespace nt2 { namespace details
{
  template<typename Shape, typename A0, typename A1> struct conv_size
  {
    typedef _2D result_type;

    static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
    {
      // We use .size() as it is defined in both
      // Expression and ConvolutionOperator concept
      std::ptrdiff_t n = a0.size()+a1.size()-1;
      return a0.extent()[0] == 1 ? result_type(1,n) : result_type(n);
    }
  };

  template<typename A0, typename A1>
  struct conv_size< policy<ext::same_>, A0, A1 >
  {
    typedef typename A0::extent_type                      result_type;

    static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
    {
      return a0.extent();
    }
  };

  template<typename A0, typename A1>
  struct conv_size< policy<ext::valid_>, A0, A1 >
  {
    typedef _2D result_type;

    static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
    {
      std::ptrdiff_t l0 = a0.size();
      std::ptrdiff_t l1 = a1.size();

      std::ptrdiff_t n  = std::max( l0 - std::max(std::ptrdiff_t(0), l1-1)
                                  , std::ptrdiff_t(0)
                                  );

      return a0.extent()[0] == 1 ? result_type(1,n) : result_type(n);
    }
  };
} }

#endif
