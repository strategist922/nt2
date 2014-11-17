//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_ADD_SETTINGS_HPP_INCLUDED
#define NT2_CORE_SETTINGS_ADD_SETTINGS_HPP_INCLUDED

#include <nt2/core/container/dsl/forward.hpp>
#include <boost/proto/expr.hpp>

namespace nt2
{
  struct settings;

  namespace meta
  {
    /*!
      @brief

      @tparam Original
      @tparam New
    **/
    template<typename Original, typename New = nt2::settings()>
    struct add_settings
    {
      using type =  nt2::settings(New,Original);
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T>
    struct add_settings<T, nt2::settings()>
    {
      using type = T;
    };

    /// INTERNAL ONLY
    template<typename T,typename U>
    struct add_settings<T, nt2::settings(U)>
    {
      using type = nt2::settings(U,T);
    };

    /// INTERNAL ONLY
    template<typename T,typename U, typename... Us>
    struct add_settings<T, nt2::settings(U,Us...)>
    {
      using type = nt2::settings(U,Us...,T);
    };

    //--------------------------------------------------------------------------
    template<typename U>
    struct add_settings<nt2::settings(),U>
    {
      using type = U;
    };

    /// INTERNAL ONLY
    template<>
    struct add_settings<nt2::settings(), nt2::settings()>
    {
      using type = nt2::settings();
    };

    /// INTERNAL ONLY
    template<typename U>
    struct add_settings<nt2::settings(), nt2::settings(U)>
    {
      using type = U;
    };

    /// INTERNAL ONLY
    template<typename U, typename... Us>
    struct add_settings<nt2::settings(), nt2::settings(U,Us...)>
    {
      using type = nt2::settings(U,Us...);
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T>
    struct add_settings<nt2::settings(T), nt2::settings()>
    {
      using type = T;
    };

    /// INTERNAL ONLY
    template<typename T, typename U>
    struct add_settings<nt2::settings(T), U>
    {
      using type = nt2::settings(U,T);
    };

    /// INTERNAL ONLY
    template<typename T,typename U>
    struct add_settings<nt2::settings(T), nt2::settings(U)>
    {
      using type = nt2::settings(U,T);
    };

    /// INTERNAL ONLY
    template<typename T,typename U, typename... Us>
    struct add_settings<nt2::settings(T), nt2::settings(U,Us...)>
    {
      using type = nt2::settings(U,Us...,T);
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T, typename... Ts>
    struct add_settings<nt2::settings(T,Ts...), nt2::settings()>
    {
      using type = nt2::settings(T,Ts...);
    };

    /// INTERNAL ONLY
    template<typename T, typename... Ts,typename U>
    struct add_settings<nt2::settings(T,Ts...), U>
    {
      using type = nt2::settings(U,T,Ts...);
    };

    /// INTERNAL ONLY
    template<typename T, typename... Ts,typename U>
    struct add_settings<nt2::settings(T,Ts...), nt2::settings(U)>
    {
      using type = nt2::settings(U,T,Ts...);
    };

    /// INTERNAL ONLY
    template<typename T, typename... Ts,typename U, typename... Us>
    struct add_settings<nt2::settings(T,Ts...), nt2::settings(U,Us...)>
    {
      using type = nt2::settings(U,Us...,T,Ts...);
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename Original, typename New>
    struct add_settings<Original&, New>
    {
      using type = typename add_settings<Original,New>::type&;
    };

    /// INTERNAL ONLY
    template<typename Original, typename New>
    struct add_settings<Original const, New>
    {
      using type = typename add_settings<Original,New>::type const;
    };

    /// INTERNAL ONLY
    template<typename Tag, typename Original, typename New>
    struct add_settings < boost::proto::basic_expr< Tag
                                                  , boost::proto::term<Original>
                                                  , 0l
                                                  >
                        , New
                        >
    {
      using new_settings = typename add_settings<Original, New>::type;
      using type = boost::proto::basic_expr < Tag
                                            , boost::proto::term <new_settings>
                                            , 0l
                                            >;
    };

    /// INTERNAL ONLY
    template<typename Expr, typename Semantic, typename New>
    struct add_settings< nt2::container::expression<Expr, Semantic>, New >
    {
      using new_expr  = typename add_settings<Expr, New>::type;
      using new_sema  = typename add_settings<Semantic, New>::type;
      using type      = nt2::container::expression<new_expr, new_sema>;
    };
  }
}

#endif
