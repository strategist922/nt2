//==============================================================================
//         Copyright 2012 - 2013 MetaScale SAS
//         Copyright 2013        Domagoj Saric, Little Endian Ltd.
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_SDK_ENFORCE_SEMANTICS_HPP_INCLUDED
#define BOOST_SIMD_SDK_ENFORCE_SEMANTICS_HPP_INCLUDED

#ifdef _MSC_VER
  // msdn.microsoft.com/en-us/library/45ec64h6.aspx
  #define BOOST_SIMD_ENFORCE_SEMANTICS_BEGIN()                                 \
      __pragma( float_control( push ) )                                        \
      __pragma( float_control( precise, on ) )                                 \
  /**/

  #define BOOST_SIMD_ENFORCE_SEMANTICS_END()                                   \
      __pragma( float_control( pop ) )                                         \
  /**/

#elif ( ( ( __GNUC__ * 10 ) + __GNUC_MINOR__ ) >= 44 )

  // http://lists.cs.uiuc.edu/pipermail/llvmdev/2013-April/061527.html
  // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=50782
  #define BOOST_SIMD_ENFORCE_SEMANTICS_BEGIN()                                 \
    _Pragma("GCC push_options")                                                \
    _Pragma("GCC optimize ( \"no-associative-math\" )")                       \
  /**/

  #define BOOST_SIMD_ENFORCE_SEMANTICS_END() _Pragma("GCC pop_options")

#else
  #define BOOST_SIMD_ENFORCE_SEMANTICS_BEGIN()
  #define BOOST_SIMD_ENFORCE_SEMANTICS_END()
#endif

#endif
