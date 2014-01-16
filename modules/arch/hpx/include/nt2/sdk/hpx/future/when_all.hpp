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
#ifndef NT2_SDK_HPX_FUTURE_WHEN_ALL_HPP_INCLUDED
#define NT2_SDK_HPX_FUTURE_WHEN_ALL_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/hpx/future/details/hpx_future.hpp>

namespace nt2
{

    namespace details
    {
        struct empty_body
        {
            int operator()()
            {
                return 0;
            }
        };
    }

    template<class Site>
    struct when_all_impl< tag::hpx_<Site> >
    {
#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 1, BOOST_DISPATCH_MAX_ARITY, \
"nt2/sdk/hpx/future/when_all.hpp") \
)

#include BOOST_PP_ITERATE()
    };
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define POINT(a,b) a.b

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) details::hpx_future<A##n> const & a##n
#define NT2_FUTURE_FORWARD_ARGS1(z,n,t) POINT(a##n,f_)


        template< BOOST_PP_ENUM_PARAMS(N, typename A) >
        hpx::lcos::unique_future<int>
        call( BOOST_PP_ENUM(N, NT2_FUTURE_FORWARD_ARGS, ~))
        {
            return details::hpx_future<int>(
              hpx::lcos::local::dataflow( \
                hpx::util::unwrapped(details::empty_body()) \
                BOOST_PP_COMMA_IF(N) \
                BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS1, ~) \
                )
             );
        }

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS1

#endif
