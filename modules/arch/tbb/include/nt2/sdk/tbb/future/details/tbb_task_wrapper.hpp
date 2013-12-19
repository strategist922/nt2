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
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_TBB_TASK_WRAPPER_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_TBB_TASK_WRAPPER_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>

#include <boost/move/move.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

namespace nt2
{
  namespace details
  {
#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 0, BOOST_DISPATCH_MAX_ARITY, \
"nt2/sdk/tbb/future/details/tbb_task_wrapper.hpp") \
)

#include BOOST_PP_ITERATE()
  }
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) BOOST_FWD_REF(A##n) a##n
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) boost::forward<A##n>(a##n)
#define NT2_FUTURE_FORWARD_ARGS3(z,n,t) a##n_(a##n)
#define NT2_FUTURE_FORWARD_ARGS4(z,n,t) A##n & a##n;

    template<class F,
             typename result_type,
             BOOST_PP_COMMA_IF(N)
             BOOST_PP_ENUM_PARAMS(N, typename A) >
    struct BOOST_PP_CAT(tbb_task_wrapper,N): public tbb::task
    {
      tbb_task_wrapper(BOOST_FWD_REF(F) f,
                       BOOST_FWD_REF(result_type) res
                       BOOST_PP_COMMA_IF(N)
                       BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~)
                      )
      : f_(f),res_(res)
        BOOST_PP_COMMA_IF(N)
        BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS3, ~)
      {}

      tbb::task* execute()
      {
        res_ = boost::move(
                f_( BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~))
                );
        return NULL;
      }

      private:

      BOOST_PP_CAT(tbb_task_wrapper,N) &
        operator=(BOOST_PP_CAT(tbb_task_wrapper,N) const&);

      F & f_;
      result_type & res_;
      BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS4, ~)
    };

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS2
#undef NT2_FUTURE_FORWARD_ARGS3
#undef NT2_FUTURE_FORWARD_ARGS4

#endif
