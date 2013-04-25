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
  //BOOST_DISPATCH_HIERARCHY_CLASS(sse_, simd_);
  struct sse_ : simd_
  {
    typedef simd__<sse_> base;
    typedef simd_ parent;
  };

  #define BOOST_SIMD_EXTENSION_CLASS(Current, Parent, Base)                    \
  struct Current : simd__<Parent>                                              \
  {                                                                            \
    typedef simd__<Base> base;                                                 \
    typedef simd__<Parent> parent;                                             \
  };                                                                           \
  /**/

  BOOST_SIMD_EXTENSION_CLASS(sse2_, sse_, sse_);
  BOOST_SIMD_EXTENSION_CLASS(sse3_, sse2_, sse_);
  BOOST_SIMD_EXTENSION_CLASS(sse4a_, sse3_, sse_);
#ifdef BOOST_SIMD_ARCH_AMD
  BOOST_SIMD_EXTENSION_CLASS(ssse3_, sse4a_, sse_);
#else
  BOOST_SIMD_EXTENSION_CLASS(ssse3_, sse3_, sse_);
#endif
  BOOST_SIMD_EXTENSION_CLASS(sse4_1_, ssse3_, sse_);
  BOOST_SIMD_EXTENSION_CLASS(sse4_2_, sse4_1_, sse_);
  BOOST_SIMD_EXTENSION_CLASS(avx_, sse4_2_, avx_);
  BOOST_SIMD_EXTENSION_CLASS(fma4_, avx_, avx_);
  BOOST_SIMD_EXTENSION_CLASS(xop_, fma4_, avx_);

  // Tag hierarchy for larrabee extensions
  BOOST_DISPATCH_HIERARCHY_CLASS(lrb_, simd_);
} } }

#endif
