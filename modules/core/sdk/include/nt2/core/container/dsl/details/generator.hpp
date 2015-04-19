//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_DSL_DETAILS_GENERATOR_HPP_INCLUDED
#define NT2_CORE_CONTAINER_DSL_DETAILS_GENERATOR_HPP_INCLUDED

#include <nt2/sdk/memory/forward/container.hpp>
#include <nt2/core/container/dsl/forward.hpp>
#include <nt2/core/container/dsl/value_type.hpp>
//#include <nt2/core/container/dsl/shape_of.hpp>
//#include <nt2/core/container/dsl/index_of.hpp>
#include <nt2/core/container/dsl/kind_of.hpp>
#include <nt2/core/functions/extent.hpp>
#include <nt2/core/utility/of_size.hpp>

#include <boost/dispatch/meta/transfer_qualifiers.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>
#include <type_traits>

namespace nt2 { namespace details
{
  //==========================================================================
  /*!
   * This metafunction specify the way a given expression is build when
   * entering a proto::generator. By default, all NT2 expressions are built
   * the following way:
   *
   *  - the value_type of the expression is computed
   *  - the size of the expression is computed
   *  - the shape of the expression is computed
   *  - the base indices of the expression is computed
   *  - if the size is _0D, then the expression will behave as its value_type
   *    else, it will behave as a container of the domain with proper shape,size
   *    and base indices.
   *
   * cref qualifiers and other peculiarities of type are conserved all along
   * so the type is actually the most optimized possible.
   *
   * \tparam Tag    Top most tag of the expression
   * \tparam Domain Domain of the expression
   * \tparam Arity  Number of children of the expression
   * \tparam Expr   The expression itself
   *
  **/
  //==========================================================================
  template<class Tag, class Domain, int Arity, class Expr> struct generator
  {
    using value_t = ext::value_type<Tag, Domain, Arity, Expr>;

    typedef typename boost::mpl::
    eval_if < boost::is_same< typename std::decay<nt2::extent_t<Expr>>::type, _0D >
            , value_t
            , boost::dispatch::meta::
              transfer_qualifiers
                    < memory::container < typename meta::kind_of<Expr>::type
                                        , typename std::decay<typename value_t::type>::type
                                        , nt2::settings()
                                        >
                    , typename value_t::type
                    >
            >::type                                               type;

    using result_type = container::expression< typename boost::remove_const<Expr>::type,type>;

    BOOST_FORCEINLINE result_type operator()(Expr& e) const { return result_type(e); }
  };
} }

#endif
