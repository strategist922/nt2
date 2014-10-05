//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_DSL_CATEGORY_HPP_INCLUDED
#define BOOST_DISPATCH_DSL_CATEGORY_HPP_INCLUDED

#include <boost/dispatch/meta/hierarchy_of.hpp>
#include <boost/dispatch/dsl/semantic_of.hpp>
#include <boost/dispatch/meta/value_of.hpp>
#include <boost/dispatch/dsl/details/expr_of.hpp>

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief Boost.Proto abstract syntax tree hierarchy

    ast_ represents any Boost.Proto expression in a given @c Domain.

    @par Models:
    Hierarchy

    @tparam Type    Hierarchized type
    @tparam Domain  Proto Domain of the categorized type
  **/
  template<typename Type, typename Domain>
  struct  ast_
#if !defined(DOXYGEN_ONLY)
        : unspecified_<Type>
#endif
  {
    /// @brief Parent hierarchy
    typedef unspecified_<Type> parent;
  };

  /*!
    @brief Boost.Proto node hierarchy

    Represents a Boost.Proto expression node by using its of tag, semantic
    , number of children and domain.

    Parent hierarchy of @c node_ is computed by using the parent hierarchy of
    @c Tag. Once reaching @c unspecified_, @c node_ parent is computed as
    @c ast_.

    @par Models:
    Hierarchy

    @tparam Semantic  Expression semantic hierarchy
    @tparam Tag       Expresson node tag hierarchy
    @tparam Arity     Expression arity
    @tparam Domain    Proto Domain of the categorized type
  **/
  template<typename Semantic, typename Tag, typename Arity, typename Domain>
  struct  node_
#if !defined(DOXYGEN_ONLY)
        : node_<Semantic, typename Tag::parent, Arity, Domain>
#endif
  {
    /// @brief Parent hierarchy
    typedef node_<Semantic, typename Tag::parent, Arity, Domain> parent;
  };

#if !defined(DOXYGEN_ONLY)
  template<typename T, typename Tag, typename N, typename D>
  struct node_<T, unspecified_<Tag>, N, D> : ast_<T, D>
  {
    typedef ast_<T, D> parent;
  };
#endif

  /*!
    @brief Boost.Proto expression hierarchy

    Represents a Boost.Proto expression node by using its of tag, semantic
    and number of children.

    Parent hierarchy of expr_ is computed by using the parent hierarchy of
    @c Semantic. Once reaching @c unspecified_, @c expr_ parent is computed as a
    @c node_ hierarchy.

    @par Models:
    Hierarchy

    @tparam Semantic  Expression semantic hierarchy
    @tparam Tag       Expresson node tag hierarchy
    @tparam Arity     Expression arity
  **/
  template<typename Semantic, typename Tag, typename Arity>
  struct  expr_
#if !defined(DOXYGEN_ONLY)
        : expr_<typename Semantic::parent, Tag, Arity>
#endif
  {
    typedef expr_<typename Semantic::parent, Tag, Arity>  parent;
  };

#if !defined(DOXYGEN_ONLY)
  template<typename T, typename Tag, typename N>
  struct  expr_< unspecified_<T>, Tag, N>
        : node_<T, Tag, N, typename details::expr_of<T>::type::proto_domain>
  {
    typedef node_<T, Tag, N, typename details::expr_of<T>::type::proto_domain> parent;
  };
#endif

} } }

namespace boost { namespace dispatch { namespace details
{
  // Proto expression hierarchy computation
  template<typename T, typename Origin>
  struct hierarchy_of< T
                     , Origin
                     , typename T::proto_is_expr_
                     >
  {
    typedef typename meta::semantic_of<T>::type  semantic_type;

    typedef meta::expr_ < typename meta::hierarchy_of<semantic_type, Origin>::type
                        , typename meta::hierarchy_of<typename T::proto_tag>::type
                        , typename T::proto_arity
                        >                                          type;
  };

  template<typename T>
  struct value_of< T
                 , typename T::proto_is_expr_
                 >
    : meta::semantic_of<T>
  {
  };

  template<typename T>
  struct value_of_cv< T const
                    , typename T::proto_is_expr_
                    >
    : meta::semantic_of<T const>
  {
  };

  template<typename T>
  struct value_of_cv< T&
                    , typename T::proto_is_expr_
                    >
    : meta::semantic_of<T&>
  {
  };
} } }

#endif
