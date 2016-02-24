//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SCALAR_GLOBAL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SCALAR_GLOBAL_HPP_INCLUDED

#include <nt2/core/functions/global.hpp>
#include <nt2/core/container/dsl.hpp>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT( global_, tag::cpu_
                          , (A0)(A1)
                          , (unspecified_<A0>)
                            ((ast_<A1, nt2::container::domain>))
                          )
  {
    typedef typename A1::value_type                                     value_type;
    typedef typename meta::result_of<A0 const(const value_type&)>::type result_type;
    typedef nt2::functor<typename A0::tag_type::neutral_element>        neutral;
    typedef nt2::functor<typename A0::tag_type::binary_op>              binary_op;

    BOOST_FORCEINLINE result_type operator()(A0 const&, A1 const& a1) const
    {
      binary_op   op;
      std::size_t sz  = numel(a1);

      result_type that = neutral()(meta::as_<result_type>());

      for(std::size_t i=0;i!=sz;++i)
        that = op(that, run(a1,i, meta::as_<result_type>()) );

      return that;
    }
  };
} }

#endif
