//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SIMD_GLOBAL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIMD_GLOBAL_HPP_INCLUDED
#ifndef BOOST_SIMD_NO_SIMD

#include <nt2/core/functions/global.hpp>
#include <nt2/include/functions/run.hpp>
#include <nt2/include/functions/sum.hpp>
#include <nt2/core/container/dsl.hpp>
#include <nt2/include/functions/numel.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/sdk/simd/meta/is_vectorizable.hpp>
#include <boost/simd/sdk/simd/category.hpp>

#include <boost/dispatch/meta/hierarchy_of.hpp>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT_IF ( global_, boost::simd::tag::simd_
                              , (A0)(A1)
                              , (boost::simd::meta::is_vectorizable
                                < typename meta::result_of< A0 const(const typename A1::value_type&)>::type
                                , BOOST_SIMD_DEFAULT_EXTENSION
                                >
                                )
                              , (unspecified_<A0>)
                                ((ast_<A1, nt2::container::domain>))
                              )
  {
    typedef typename A1::value_type                                         value_type;
    typedef typename meta::result_of<A0 const(const value_type&)>::type     result_type;
    typedef nt2::functor<typename A0::tag_type::neutral_element>            neutral;
    typedef nt2::functor<typename A0::tag_type::binary_op>                  binary_op;
    typedef boost::simd::native<result_type, BOOST_SIMD_DEFAULT_EXTENSION>  target_type;
    typedef meta::as_elementwise<A1>                                        sched;

    BOOST_FORCEINLINE result_type operator()(A0 const& a0, A1 const& a1) const
    {
      static const std::size_t N = boost::simd::meta::cardinal_of<target_type>::value;
      static const std::size_t M = ~(N-1);

      std::size_t sz          = numel(a1);
      std::size_t aligned_sz  = sz & M;
      std::size_t i;

      target_type vthat = neutral()(meta::as_<target_type>());
      binary_op op;

      auto x = sched::call(a1);

      for(i=0;i!=aligned_sz;i+=N)
        vthat = op(vthat, nt2::run(x,i, meta::as_<target_type>()) );

      result_type that = a0(vthat);

      for(;i!=sz;++i)
        that = op(that, nt2::run(x,i, meta::as_<result_type>()) );

      return that;
    }
  };
} }

#endif

#endif
