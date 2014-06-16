//==============================================================================
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_VIEW_ADAPTED_VIEW_TYPE_HPP_INCLUDED
#define NT2_CORE_CONTAINER_VIEW_ADAPTED_VIEW_TYPE_HPP_INCLUDED

#include <nt2/sdk/memory/container_ref.hpp>
#include <nt2/sdk/meta/container_traits.hpp>
#include <boost/config.hpp>

namespace nt2 { namespace container
{
  template<class Expression, class ResultType> struct expression;
} }

namespace nt2 { namespace details
{
  template<typename Container>
  struct view_type
  {
    // Base container type
    typedef typename meta::kind_<Container>::type         kind_type;
    typedef typename meta::value_type_<Container>::type   value_type;
    typedef typename Container::settings_type             settings_type;

    typedef memory::container_ref < kind_type
                                  , value_type
                                  , settings_type
                                  >                       container_ref;

    typedef memory::container < kind_type
                              , value_type
                              , settings_type
                              >&                          container_type;

    typedef boost::proto::basic_expr< boost::proto::tag::terminal
                                    , boost::proto::term<container_ref>
                                    , 0l
                                    >                     basic_expr;

    typedef nt2::container::expression< basic_expr
                                      , container_type
                                      >                     nt2_expression;
  };

  template<typename Container>
  struct view_type<Container const>
  {
    // Base container type
    typedef typename meta::kind_<Container>::type               kind_type;
    typedef typename meta::value_type_<Container>::type         value_type;
    typedef typename Container::settings_type                   settings_type;

    typedef memory::container_ref < kind_type
                                  , value_type const
                                  , settings_type
                                  >                       container_ref;

    typedef memory::container < kind_type
                              , value_type
                              , settings_type
                              > const&                    container_type;

    typedef boost::proto::basic_expr< boost::proto::tag::terminal
                                    , boost::proto::term<container_ref>
                                    , 0l
                                    >                     basic_expr;

    typedef nt2::container::expression< basic_expr
                                      , container_type
                                      >                     nt2_expression;
  };
} }

#endif
