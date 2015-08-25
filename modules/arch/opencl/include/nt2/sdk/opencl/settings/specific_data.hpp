//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENCL_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED
#define NT2_SDK_OPENCL_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED

#ifdef NT2_HAS_OPENCL

#include <nt2/sdk/opencl/settings/specific_opencl.hpp>
#include <nt2/core/settings/specific_data.hpp>
#include <nt2/sdk/opencl/opencl.hpp>

namespace nt2{

  namespace tag{
    template<typename Site>
      struct opencl_;
  }

  template<typename Site, typename T>
  struct specific_data< tag::opencl_<Site>,T >
  {
    using type = details::specific_opencl<tag::opencl_<Site>,T >;
  };

}


#endif
#endif
