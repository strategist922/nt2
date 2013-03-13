//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_SDK_SIMD_ALGORITHM_HPP_INCLUDED
#define BOOST_SIMD_SDK_SIMD_ALGORITHM_HPP_INCLUDED

#include <boost/simd/sdk/simd/native.hpp>
#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/store.hpp>
#include <boost/simd/include/functions/unaligned_load.hpp>
#include <boost/simd/include/functions/splat.hpp>
#include <boost/simd/sdk/memory/align_on.hpp>
#include <boost/simd/sdk/simd/meta/unroll.hpp>
#include <boost/mpl/assert.hpp>

namespace boost { namespace simd
{
  namespace detail
  {
    template<class T, class U, class vT, class vU, class F>
    struct transform_unary_impl
    {
      BOOST_FORCEINLINE
      transform_unary_impl(T const* begin_, U* out_, F& f_) : begin(begin_), out(out_), f(f_)
      {
      }

      BOOST_FORCEINLINE
      void operator()(std::size_t i) const
      {
        simd::store(f(simd::unaligned_load<vT>(begin, i)), out, i);
      }

      T const* begin;
      U* out;
      F& f;
    };
  }

  template<class T, class U, class UnOp>
  U* transform(T const* begin, T const* end, U* out, UnOp f)
  {
    typedef boost::simd::native<T, BOOST_SIMD_DEFAULT_EXTENSION> vT;
    typedef boost::simd::native<U, BOOST_SIMD_DEFAULT_EXTENSION> vU;

    BOOST_MPL_ASSERT_MSG( vT::static_size == vU::static_size
                        , BOOST_SIMD_TRANSFORM_INPUT_OUTPUT_NOT_SAME_SIZE
                        , (T, U)
                        );

    static const std::size_t N = vU::static_size;

    std::size_t shift = simd::align_on(out, N * sizeof(U)) - out;
    T const* end2 = begin + shift;
    std::size_t iter_size = (end - end2)/N*N;

    // prologue
    for(; begin!=end2; ++begin, ++out)
      *out = f(*begin);

    meta::unroll<16, N>::apply(0, iter_size, detail::transform_unary_impl<T, U, vT, vU, UnOp>(begin, out, f));
    begin += iter_size;
    out += iter_size;

    // epilogue
    for(; begin!=end; ++begin, ++out)
      *out = f(*begin);

    return out;
  }

  namespace detail
  {
    template<class T1, class T2, class U, class vT1, class vT2, class vU, class F>
    struct transform_binary_impl
    {
      BOOST_FORCEINLINE
      transform_binary_impl(T1 const* begin1_, T2 const* begin2_, U* out_, F& f_) : begin1(begin1_), begin2(begin2_), out(out_), f(f_)
      {
      }

      BOOST_FORCEINLINE
      void operator()(std::size_t i) const
      {
        simd::store(f(simd::unaligned_load<vT1>(begin1, i), simd::unaligned_load<vT2>(begin2, i)), out, i);
      }

      T1 const* begin1;
      T2 const* begin2;
      U* out;
      F& f;
    };
  }

  template<class T1, class T2, class U, class BinOp>
  U* transform(T1 const* begin1, T1 const* end, T2 const* begin2, U* out, BinOp f)
  {
    typedef boost::simd::native<T1, BOOST_SIMD_DEFAULT_EXTENSION> vT1;
    typedef boost::simd::native<T2, BOOST_SIMD_DEFAULT_EXTENSION> vT2;
    typedef boost::simd::native<U, BOOST_SIMD_DEFAULT_EXTENSION> vU;

    BOOST_MPL_ASSERT_MSG( vT1::static_size == vT2::static_size && vT1::static_size == vU::static_size
                        , BOOST_SIMD_TRANSFORM_INPUT_OUTPUT_NOT_SAME_SIZE
                        , (T1, T2, U)
                        );

    static const std::size_t N = vU::static_size;

    std::size_t shift = simd::align_on(out, N * sizeof(U)) - out;
    T1 const* end2 = begin1 + shift;
    std::size_t iter_size = (end - end2)/N*N;

    // prologue
    for(; begin1!=end2; ++begin1, ++begin2, ++out)
      *out = f(*begin1, *begin2);

    meta::unroll<16, N>::apply(0, iter_size, detail::transform_binary_impl<T1, T2, U, vT1, vT2, vU, BinOp>(begin1, begin2, out, f));
    begin1 += iter_size;
    begin2 += iter_size;
    out += iter_size;

    // epilogue
    for(; begin1!=end; ++begin1, ++begin2, ++out)
      *out = f(*begin1, *begin2);

    return out;
  }

  namespace detail
  {
    template<class T, class vT, class vU, class F>
    struct accumulate_impl
    {
      BOOST_FORCEINLINE
      accumulate_impl(T const* begin_, vU& cur_, F& f_) : begin(begin_), cur(cur_), f(f_)
      {
      }

      BOOST_FORCEINLINE
      void operator()(std::size_t i) const
      {
        cur = f(cur, boost::simd::load<vT>(begin, i));
      }

      T const* begin;
      vU& cur;
      F& f;
    };
  }

  template<class T, class U, class F>
  U accumulate(T const* begin, T const* end, U init, F f)
  {
    typedef boost::simd::native<T, BOOST_SIMD_DEFAULT_EXTENSION> vT;
    typedef boost::simd::native<U, BOOST_SIMD_DEFAULT_EXTENSION> vU;

    BOOST_MPL_ASSERT_MSG( vT::static_size == vU::static_size
                        , BOOST_SIMD_ACCUMULATE_INPUT_OUTPUT_NOT_SAME_SIZE
                        , (T, U)
                        );

    static const std::size_t N = vT::static_size;

    T const* end2 = simd::align_on(begin, N * sizeof(T));
    std::size_t iter_size = (end - end2)/N*N;

    vU cur = simd::splat<vU>(init);

    // prologue
    for(; begin!=end2; ++begin)
      init = f(init, *begin);

    meta::unroll<16, N>::apply(0, iter_size, detail::accumulate_impl<T, vT, vU, F>(begin, cur, f));
    begin += iter_size;

    // reduce cur
    for(U const* b = cur.begin(); b != cur.end(); ++b)
      init = f(init, *b);

    // epilogue
    for(; begin!=end; ++begin)
      init = f(init, *begin);

    return init;
  }
} }

#endif
