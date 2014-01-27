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
#ifndef NT2_SDK_OPENMP_FUTURE_WHEN_ALL_HPP_INCLUDED
#define NT2_SDK_OPENMP_FUTURE_WHEN_ALL_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 201307 /* OpenMP 4.0 */

#include <omp.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

#include <nt2/sdk/openmp/future/details/openmp_future.hpp>

namespace nt2
{
    template<class Site>
    struct when_all_impl< tag::openmp_<Site> >
    {

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 1, BOOST_DISPATCH_MAX_ARITY, \
"nt2/sdk/openmp/future/when_all.hpp") \
)

#include BOOST_PP_ITERATE()
    };
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define POINT(a,b) a.b

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) details::openmp_future<A##n> const & a##n
#define NT2_FUTURE_FORWARD_ARGS1(z,n,t) A##n & r##n = *(POINT(a##n,ready_));
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) r##n

template< BOOST_PP_ENUM_PARAMS(N, typename A) >
details::openmp_future<int> call\
( BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~))
{
    details::openmp_future<int> future_res;

    bool & next( *(future_res.ready_) );

    BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS1, ~)

    #pragma omp task \
       shared( next, BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~)) \
       depend( in : BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~) ) \
       depend( out : next )
    {
        *(future_res.res_) = 0;
        next = true;
    }

    return future_res;
}

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS1
#undef NT2_FUTURE_FORWARD_ARGS2

#endif
