//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_META_LOCALITY_HPP_INCLUDED
#define NT2_SDK_META_LOCALITY_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/core/settings/forward/locality.hpp>
#include <nt2/core/settings/option.hpp>

namespace nt2{

  template<typename X>
  typename meta::option<X,tag::locality_>::type  locality(X const&)
  {
    return {};
  }

}


#endif
#endif




