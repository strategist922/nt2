//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_OPTIONS_HPP_INCLUDED
#define NT2_SIGNAL_OPTIONS_HPP_INCLUDED

#include <nt2/sdk/meta/policy.hpp>

namespace nt2
{
  namespace ext
  {
    struct full_  {};
    struct same_  {};
    struct valid_ {};
  }

  policy<ext::full_>  const full_;
  policy<ext::same_>  const same_;
  policy<ext::valid_> const valid_;
}

#endif
