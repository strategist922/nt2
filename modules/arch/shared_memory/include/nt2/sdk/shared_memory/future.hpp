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
#ifndef NT2_SDK_SHARED_MEMORY_FUTURE_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_FUTURE_HPP_INCLUDED

#include <boost/move/move.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/dispatch/details/parameters.hpp>
#include <boost/utility/result_of.hpp>

namespace nt2
{
    template<class Arch, class result_type>
    struct make_future;

    template<class Arch>
    struct async_impl;

    template<class Arch>
    struct when_all_impl;

    template<class Arch>
    struct make_ready_future_impl;

    template< typename Arch, typename result_type>
    inline typename make_future< Arch,result_type>::type
    make_ready_future(result_type const & value)
    {
       return make_ready_future_impl<Arch>().call(value);
    }

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 0, BOOST_DISPATCH_MAX_ARITY, "nt2/sdk/shared_memory/future.hpp")\
)
#include BOOST_PP_ITERATE()

}

#endif

#else

#define N BOOST_PP_ITERATION()

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) BOOST_FWD_REF(A##n) a##n
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) boost::forward<A##n>(a##n)
#define NT2_FUTURE_FORWARD_ARGS3(z,n,t) A##n & a##n

    template< typename Arch,typename F\
              BOOST_PP_COMMA_IF(N)\
              BOOST_PP_ENUM_PARAMS(N, typename A)\
          >
    inline typename make_future< Arch,\
                       typename boost::result_of<\
                       F(BOOST_PP_ENUM_PARAMS(N, A))\
                       >::type\
                       >::type
    async( BOOST_FWD_REF(F) f\
           BOOST_PP_COMMA_IF(N)\
           BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~)\
       )
    {
        return async_impl<Arch>().call(boost::forward<F>(f) \
                                       BOOST_PP_COMMA_IF(N)\
                                       BOOST_PP_ENUM(N,\
                                       NT2_FUTURE_FORWARD_ARGS2, ~)\
                                      );
    }

#if BOOST_PP_GREATER(N,0)

    template< typename Arch,\
              BOOST_PP_ENUM_PARAMS(N, typename A)\
              >
    inline typename make_future< Arch,int>::type
    when_all(BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS3, ~))
    {
      return when_all_impl<Arch>().call(BOOST_PP_ENUM_PARAMS(N,a));
    }

#endif

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS2
#undef NT2_FUTURE_FORWARD_ARGS3

#endif

