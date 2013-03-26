//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013 MetaScale
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_SDK_SIMD_EXTENSIONS_META_X86_TAGS_HPP_INCLUDED
#define BOOST_SIMD_SDK_SIMD_EXTENSIONS_META_X86_TAGS_HPP_INCLUDED

#include <boost/dispatch/functor/meta/hierarchy.hpp>
#include <boost/simd/sdk/simd/extensions/meta/common/tags.hpp>
#include <boost/simd/sdk/config/arch.hpp>

namespace boost { namespace simd { namespace tag
{
  // Tag hierarchy for SSE extensions
  BOOST_DISPATCH_HIERARCHY_CLASS(sse_, simd_);
  BOOST_DISPATCH_HIERARCHY_CLASS(sse2_, simd__<sse_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(sse3_, simd__<sse2_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(sse4a_, simd__<sse3_>);
#ifdef BOOST_SIMD_ARCH_AMD
  BOOST_DISPATCH_HIERARCHY_CLASS(ssse3_, simd__<sse4a_>);
#else
  BOOST_DISPATCH_HIERARCHY_CLASS(ssse3_, simd__<sse3_>);
#endif
  BOOST_DISPATCH_HIERARCHY_CLASS(sse4_1_, simd__<ssse3_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(sse4_2_, simd__<sse4_1_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(avx_, simd__<sse4_2_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(fma4_, simd__<avx_>);
  BOOST_DISPATCH_HIERARCHY_CLASS(xop_, simd__<fma4_>);

  // Tag hierarchy for larrabee extensions
  BOOST_DISPATCH_HIERARCHY_CLASS(lrb_, simd_);
} } }

#endif
