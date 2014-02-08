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

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>

namespace nt2
{

    namespace details
    {
      struct empty_body
      {
        typedef int result_type;

        template <typename T>
        int operator()(T) const
        {
           return 0;
        }

      };
    }

    template<class Site>
    struct when_all_impl< tag::hpx_<Site> >
    {

        template <typename Future>
        inline hpx::lcos::unique_future< std::vector<Future> >
        call( BOOST_FWD_REF(std::vector<Future>) lazy_values )
        {
            return hpx::when_all( boost::forward< std::vector<Future> >(\
                                    lazy_values)\
                                ).then(details::empty_body());
        }

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

#define HPX_WAIT_ALL_FUTURE_ARG(z, n, t) hpx::lcos::unique_future<A##n> & a##n

        template< BOOST_PP_ENUM_PARAMS(N, typename A) >
        inline hpx::lcos::unique_future<int>
        call( BOOST_PP_ENUM(N, HPX_WAIT_ALL_FUTURE_ARG, ~))
        {
            return hpx::when_all(BOOST_PP_ENUM_PARAMS(N,a)).then(details::empty_body());
        }

#undef HPX_WAIT_ALL_FUTURE_ARG
#undef N

#endif
