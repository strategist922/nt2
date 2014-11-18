//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_BUFFER_HPP_INCLUDED
#define NT2_CORE_SETTINGS_BUFFER_HPP_INCLUDED

#include <nt2/core/settings/details/is_buffer.hpp>

namespace nt2 { namespace tag
{
  /// @brief Buffer option mark-up
  struct buffer_ {};

  //----------------------------------------------------------------------------
  /// INTERNAL ONLY
  template<typename T>
  typename details::is_buffer<T>::type match_(buffer_, T);
} }

#endif
