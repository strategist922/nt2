//==============================================================================
//         Copyright 2015 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_COMMON_BITWISE_SLIDE_HPP_INCLUDED
#define BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_COMMON_BITWISE_SLIDE_HPP_INCLUDED

#include <boost/simd/memory/functions/bitwise_slide.hpp>
#include <boost/simd/memory/functions/slide.hpp>
#include <boost/simd/include/functions/bitwise_cast.hpp>
#include <boost/simd/include/functions/shift_left.hpp>
#include <boost/simd/include/functions/shift_right.hpp>
#include <boost/simd/include/functions/bitwise_xor.hpp>
#include <boost/dispatch/functor/preprocessor/call.hpp>
#include <boost/dispatch/meta/mpl.hpp>
#include <boost/dispatch/meta/scalar_of.hpp>
#include <boost/dispatch/meta/as_unsigned.hpp>
#include <boost/dispatch/meta/upgrade.hpp>
#include <boost/dispatch/attributes.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost { namespace simd { namespace ext
{
  using dispatch::meta::upgrade;

  template<typename T,typename U=void>
  struct biggest
  {};

  template<typename T>
  struct biggest<T, typename boost::enable_if_c< boost::is_same<T
                  , typename upgrade<T>::type>::value >::type>
  {
    typedef T type;
  };

  template<typename T>
  struct biggest<T, typename boost::disable_if_c< boost::is_same<T
                  , typename upgrade<T>::type>::value >::type>
  {
    typedef typename biggest<typename upgrade<T>::type>::type type;
  };

  BOOST_DISPATCH_IMPLEMENT          ( bitwise_slide_
                                    , boost::simd::tag::cpu_
                                    , (A0)(N)(X)
                                    , ((simd_< integer_<A0>,X>))
                                      (mpl_integral_< scalar_< integer_<N> > >)
                                    )
  {
    typedef A0 result_type;

    typedef typename dispatch::meta::as_unsigned<A0>::type u_t;
    typedef typename dispatch::meta::scalar_of<u_t>::type s_t;
    typedef typename biggest<s_t>::type b_t;
    typedef boost::simd::native<b_t,X> n_t;

    BOOST_FORCEINLINE result_type operator()(A0 const& a0, N const&) const
    {
      return bitwise_cast<A0>(eval(boost::simd::bitwise_cast<n_t>(a0),boost::mpl::bool_<(N::value>=0)>()));
    }

    BOOST_FORCEINLINE n_t eval(n_t const& a0, boost::mpl::true_ const&) const
    {
      n_t a = shift_left(a0,N::value);
      n_t b = shift_right(a0,sizeof(b_t)*CHAR_BIT-N::value);
      n_t c = slide<1>(b);
      return bitwise_xor(a,c);
    }

    BOOST_FORCEINLINE n_t eval(n_t const& a0, boost::mpl::false_ const&) const
    {
      n_t a = shift_right(a0,-N::value);
      n_t b = shift_left(a0,sizeof(b_t)*CHAR_BIT+N::value);
      n_t c = slide<-1>(b);
      return bitwise_xor(a,c);
    }
  };
} } }

#endif
