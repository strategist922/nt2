//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_SETTINGS_HPP_INCLUDED
#define NT2_CORE_SETTINGS_SETTINGS_HPP_INCLUDED

#include <nt2/core/settings/option.hpp>
#include <nt2/core/settings/match_option.hpp>

#include <boost/mpl/deref.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/dispatch/meta/any.hpp>

namespace nt2
{
  /*!
    @brief Settings pack mark-up type

    settings is a seed type that allow construction of multiple options pack.
    If given container needs to have options A and B set, settings(A,B) will
    perform such a task.
  **/
  struct settings {};

  namespace meta
  {
    /// INTERNAL ONLY
    template< class S0, class Option>
    struct  match_option< settings(S0), Option >
          : match_option<S0,Option>
    {};

    /// INTERNAL ONLY
    template<typename Option, typename S0, typename S1, typename... S>
    struct  match_option< settings(S0,S1,S...), Option >
          : boost::dispatch::meta::any< lambda_match_option<Option>
                                      , S0, S1, S...
                                      >
    {};

    /// INTERNAL ONLY
    template<class S0, class Option, class Semantic>
    struct  option< settings(S0), Option, Semantic >
          : option<S0,Option,Semantic>
    {};

    /// INTERNAL ONLY
    template< typename Option, typename Semantic
            , typename S0, typename S1, typename... S
            >
    struct option<settings(S0,S1,S...), Option, Semantic>
    {
      typedef typename boost::mpl::find_if< boost::mpl::vector<S0,S1,S...>
                                          , lambda_match_option<Option>
                                          >::type               it_t;
      typedef typename boost::mpl::deref<it_t>::type            found_t;
      typedef typename option<found_t,Option,Semantic>::type    type;
    };
  }
}

#endif
