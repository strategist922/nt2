//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_HPX_RUNTIME_COSTS_TRANSFORM_HPP_INCLUDED
#define NT2_SDK_HPX_RUNTIME_COSTS_TRANSFORM_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <nt2/sdk/runtime_costs.hpp>
#include <nt2/sdk/shared_memory/runtime_costs.hpp>

#ifndef BOOST_NO_EXCEPTIONS
#include <boost/exception_ptr.hpp>
#endif

namespace nt2
{
  namespace tag
    {
      struct transform_;
      template<class T> struct hpx_;
    }

  template<class Site>
  struct runtime_costs< tag::transform_, tag::hpx_<Site> >
  {
    typedef hpx_transform type;
  };
}

#endif
#endif
