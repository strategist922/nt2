//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_NT2_LAUNCH_POLICY_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_NT2_LAUNCH_POLICY_HPP_INCLUDED

namespace nt2
{
 namespace launch
 {
#if defined(NT2_LAUNCH_DEFERRED)
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = (std::launch::async | std::launch::deferred);
#endif
 }
}

#endif
