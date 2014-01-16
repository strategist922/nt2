//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#if !BOOST_PP_IS_ITERATING
#ifndef NT2_SDK_HPX_FUTURE_FUTURE_HPP_INCLUDED
#define NT2_SDK_HPX_FUTURE_FUTURE_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <boost/move/move.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/hpx/future/details/hpx_future.hpp>

namespace nt2
{
    namespace tag
    {
        template<class T> struct hpx_;
    }

    template<class Site, class result_type>
    struct make_future<tag::hpx_<Site> , result_type>
    {
        typedef details::hpx_future<result_type> type;
    };

    template< class Site>
    struct make_ready_future_impl< tag::hpx_<Site> >
    {
        template< typename result_type >
        inline details::hpx_future<result_type>
        call(BOOST_FWD_REF(result_type) value)
        {
          return
            details::hpx_future<result_type>(
              hpx::make_ready_future( \
                boost::forward<result_type>(value) \
                )
              );
        }
    };

    template<class Site>
    struct async_impl< tag::hpx_<Site> >
    {
#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 0, BOOST_DISPATCH_MAX_ARITY, "nt2/sdk/hpx/future/future.hpp") \
)
#include BOOST_PP_ITERATE()
    };
}

#endif
#endif
#else

#define N BOOST_PP_ITERATION()

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) BOOST_FWD_REF(A##n) a##n

#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) boost::forward<A##n>(a##n)

        template< typename F \
          BOOST_PP_COMMA_IF(N) \
          BOOST_PP_ENUM_PARAMS(N, typename A) >
        inline typename make_future< \
          tag::hpx_<Site>,\
          typename boost::result_of< \
            F(BOOST_PP_ENUM_PARAMS(N, A)) \
            >::type \
          >::type
        call(F & f \
          BOOST_PP_COMMA_IF(N) \
          BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~)
          )
        {
            typedef typename boost::result_of< \
              F(BOOST_PP_ENUM_PARAMS(N, A)) \
              >::type result_type;

            return details::hpx_future<result_type>(
              hpx::async( f \
                BOOST_PP_COMMA_IF(N) \
                BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~) \
                )
              );
        }

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS2

#endif
