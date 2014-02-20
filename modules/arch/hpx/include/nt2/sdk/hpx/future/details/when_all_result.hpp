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
#ifndef NT2_SDK_HPX_FUTURE_DETAILS_WHEN_ALL_RESULT_HPP_INCLUDED
#define NT2_SDK_HPX_FUTURE_DETAILS_WHEN_ALL_RESULT_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>

#include <vector>

namespace nt2
{
    template<typename Site, class Future>
    struct when_all_vec_result< tag::hpx_<Site>,Future>
    {
       typedef typename \
         hpx::lcos::unique_future< \
           std::vector<Future> \
         > type;
    };

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 1, BOOST_DISPATCH_MAX_ARITY, \
"nt2/sdk/hpx/future/details/when_all_result.hpp") \
)

#include BOOST_PP_ITERATE()
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()
#define HPX_WHEN_N_DECAY_FUTURE(z, n, t) typename hpx::util::decay<A##n>::type


    template< typename Site,\
          BOOST_PP_ENUM_PARAMS(N, typename A)\
          >
    struct BOOST_PP_CAT(when_all_result,N) < \
      tag::hpx_<Site>,BOOST_PP_ENUM_PARAMS(N,A) \
      >
    {
        typedef typename hpx::lcos::unique_future< \
          HPX_STD_TUPLE< \
            BOOST_PP_ENUM(N, HPX_WHEN_N_DECAY_FUTURE, ~) \
          > \
        > type;
    };

#undef HPX_WHEN_N_DECAY_FUTURE
#undef N

#endif

