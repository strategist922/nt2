//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_SDK_SIMD_PACK_DOMAIN_HPP_INCLUDED
#define BOOST_SIMD_SDK_SIMD_PACK_DOMAIN_HPP_INCLUDED

#include <boost/simd/sdk/simd/pack/forward.hpp>
#include <boost/proto/domain.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd
{
  ////////////////////////////////////////////////////////////////////////////
  // Tell proto that in the simd::domain, all expressions should be
  // wrapped in simd::expr<> using simd::generator
  ////////////////////////////////////////////////////////////////////////////
  struct domain : boost::proto::domain< generator
                                      , grammar
                                      >
  {
    template<class T, class Dummy = void>
    struct as_child : boost::proto::callable
    {
      typedef boost::proto::basic_expr< boost::proto::tag::terminal, boost::proto::term<T&> > expr_t;
      typedef expression<expr_t, T&> result_type;
      BOOST_FORCEINLINE result_type operator()(T& t) const
      {
        result_type that = { expr_t::make(t) };
        return that;
      }
    };

    template<class T, class Tag>
    struct as_child_expr : boost::proto::domain<generator, grammar>::template as_child<T>
    {
    };

    template<class T>
    struct as_child<T, typename T::proto_is_expr_>
         : as_child_expr<T, typename T::proto_tag> {};
  };
} }

#endif
