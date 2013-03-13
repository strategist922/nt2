//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_SDK_SIMD_META_UNROLL_HPP_INCLUDED
#define BOOST_SIMD_SDK_SIMD_META_UNROLL_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////
// Specializations for unroll (Duff's devices optimization)
////////////////////////////////////////////////////////////////////////////////

namespace boost { namespace simd { namespace meta
{
  template<int N, std::size_t Step = 1>
  struct unroll
  {};

  template<std::size_t N>
  struct unroll<0, N>
  {
    template<class F>
    static void
    apply( std::size_t begin, std::size_t end, F f )
    {
      for(; begin != end; begin += N)
        f(begin);
    }
  };

  template<std::size_t N>
  struct unroll<2, N>
  {
    template<class F>
    static void
    apply( std::size_t begin, std::size_t end, F f )
    {
      std::size_t distance = (end - begin)/N;
      std::size_t n = (distance + 1) / 2;

      switch(distance % 2)
      {
        case 0 : do {
          f(begin); begin += N;
        case 1 : f(begin); begin += N;
        } while(--n != 0);
      }
    }
  };

  template<std::size_t N>
  struct unroll<4, N>
  {
    template<class F>
    static void
    apply( std::size_t begin, std::size_t end, F f )
    {
      std::size_t distance = (end - begin)/N;
      std::size_t n = (distance + 3) / 4;

      switch(distance % 4)
      {
        case 0 : do {
          f(begin); begin += N;
        case 3 : f(begin); begin += N;
        case 2 : f(begin); begin += N;
        case 1 : f(begin); begin += N;
        } while(--n != 0);
      }
    }
  };

  template<std::size_t N>
  struct unroll<8, N>
  {
    template<class F>
    static void
    apply( std::size_t begin, std::size_t end, F f )
    {
      std::size_t distance = (end - begin)/N;
      std::size_t n = (distance + 7) / 8;

      switch(distance % 8)
      {
        case 0 : do {
          f(begin); begin += N;
        case 7 : f(begin); begin += N;
        case 6 : f(begin); begin += N;
        case 5 : f(begin); begin += N;
        case 4 : f(begin); begin += N;
        case 3 : f(begin); begin += N;
        case 2 : f(begin); begin += N;
        case 1 : f(begin); begin += N;
        } while(--n != 0);
      }
    }
  };

  template<std::size_t N>
  struct unroll<16, N>
  {
    template<class F>
    static void
    apply( std::size_t begin, std::size_t end, F f )
    {
      std::size_t distance = (end - begin)/N;
      std::size_t n = (distance + 15) / 16;

      switch(distance % 16)
      {
        case 0 : do {
          f(begin); begin += N;
        case 15 : f(begin); begin += N;
        case 14 : f(begin); begin += N;
        case 13 : f(begin); begin += N;
        case 12 : f(begin); begin += N;
        case 11 : f(begin); begin += N;
        case 10 : f(begin); begin += N;
        case 9  : f(begin); begin += N;
        case 8  : f(begin); begin += N;
        case 7  : f(begin); begin += N;
        case 6  : f(begin); begin += N;
        case 5  : f(begin); begin += N;
        case 4  : f(begin); begin += N;
        case 3  : f(begin); begin += N;
        case 2  : f(begin); begin += N;
        case 1  : f(begin); begin += N;
        } while(--n != 0);
      }
    }
  };

} } }



#endif
