//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_DETAILS_CONV_OFFSET_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_CONV_OFFSET_HPP_INCLUDED

#include <nt2/include/functions/length.hpp>

namespace nt2 { namespace details
{
  /*
    conv_offset computes the offset in element to apply when storing the
    result of a convolution
  */
  template<typename Filter> BOOST_FORCEINLINE
  std::size_t conv_offset(Filter const& f, policy<ext::full_> const&)
  {
    return 0;
  }

  template<typename Filter> BOOST_FORCEINLINE
  std::size_t conv_offset(Filter const& f, policy<ext::same_> const&)
  {
    return length(f)/2;
  }

  template<typename Filter> BOOST_FORCEINLINE
  std::size_t conv_offset(Filter const& f, policy<ext::valid_> const&)
  {
    return length(f)-1;
  }
} }

#endif
