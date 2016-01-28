//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_FUNCTOR_SITE_HPP_INCLUDED
#define NT2_SDK_FUNCTOR_SITE_HPP_INCLUDED

#define BOOST_SIMD_DEFINED_SITE
#include <boost/simd/sdk/simd/extensions.hpp>

#if defined(NT2_HAS_CUDA)
#include <nt2/sdk/cuda/cuda.hpp>
#else

namespace nt2
{
  template<typename Site> using accelerator_site = Site;
}

#endif

#if defined(_OPENMP)
#include <nt2/sdk/openmp/shared_memory.hpp>

#elif defined(NT2_USE_TBB)
#include <nt2/sdk/tbb/shared_memory.hpp>

#elif defined(NT2_USE_HPX)
#include <nt2/sdk/hpx/shared_memory.hpp>

#else

namespace nt2
{
  template<typename Site> using shared_memory_site = Site;
}

#endif

namespace nt2 { namespace details
{
  using final_locality_t = accelerator_site<shared_memory_site<BOOST_SIMD_DEFAULT_SITE>>;
} }

BOOST_DISPATCH_COMBINE_SITE( nt2::details::final_locality_t )

#undef BOOST_SIMD_DEFINED_SITE

#endif
