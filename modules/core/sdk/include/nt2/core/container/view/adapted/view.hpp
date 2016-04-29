//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_VIEW_ADAPTED_VIEW_HPP_INCLUDED
#define NT2_CORE_CONTAINER_VIEW_ADAPTED_VIEW_HPP_INCLUDED

#include <nt2/sdk/meta/settings_of.hpp>
#include <nt2/core/settings/option.hpp>
#include <nt2/core/settings/add_settings.hpp>
#include <nt2/core/container/dsl/forward.hpp>
#include <boost/dispatch/meta/model_of.hpp>
#include <boost/dispatch/meta/value_of.hpp>

namespace nt2 { namespace meta
{
  /// INTERNAL ONLY : Option of a view use its settings and semantic
  template<typename Container, typename Tag>
  struct  option<nt2::container::view<Container> , Tag>
        : option<typename Container::settings_type,Tag>
  {};

  /// INTERNAL ONLY : add_settinfs to a view
  template<typename Container, typename S2>
  struct add_settings< container::view<Container>, S2 >
  {
    typedef container::view<typename add_settings<Container, S2>::type> type;
  };

  /// INTERNAL ONLY : Extract settings from view
  template<typename Container>
  struct settings_of< container::view<Container> >
  {
    typedef typename Container::settings_type type;
  };
} }

namespace boost { namespace dispatch { namespace meta
{
  /// INTERNAL ONLY : value_of for view
  template<typename Container>
  struct value_of< nt2::container::view<Container> >
  {
    typedef typename nt2::container::view<Container>::value_type type;
  };

  /// INTERNAL ONLY : model_of for view
  template<typename Container>
  struct model_of< nt2::container::view<Container> >
  {
    struct type
    {
      template<class X> struct apply
      {
        typedef typename model_of<Container>::type        model_t;
        typedef typename model_t::template apply<X>::type new_t;
        typedef nt2::container::view<new_t> type;
      };
    };
  };

  /// INTERNAL ONLY : semantic_of for view
  template<typename Container>
  struct semantic_of< nt2::container::view<Container> >
  {
    typedef typename nt2::container::view<Container>::container_type  type;
  };
} } }

#endif
