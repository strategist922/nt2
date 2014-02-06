//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENMP_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED
#define NT2_SDK_OPENMP_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED

#include <nt2/sdk/openmp/shared_memory.hpp>
#include <nt2/core/settings/specific_data.hpp>
#include <nt2/sdk/shared_memory/settings/container_has_futures.hpp>

#include <boost/swap.hpp>

#include <vector>

namespace nt2
{
  template <typename Site, typename T>
  struct specific_data< tag::openmp_<Site>,T>
  {
    typedef typename
    details::container_has_futures< Arch > type;
  };
}

#endif
