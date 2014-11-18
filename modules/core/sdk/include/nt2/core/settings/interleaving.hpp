//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_INTERLEAVING_HPP_INCLUDED
#define NT2_CORE_SETTINGS_INTERLEAVING_HPP_INCLUDED

#include <boost/mpl/bool.hpp>

namespace nt2
{
  namespace details
  {
    template<bool B> struct soa_status
    {
      using interleaving_type = boost::mpl::bool_<B>;
    };
  }

  /*!
    @brief interleaved_ option

    Containers can be marked interleaved_ to express the fact that their
    contents is stored as an Array of Structure whenever needed.
  **/
  using interleaved_ = details::soa_status<false>;

  /*!
    @brief deinterleaved_ option

    Containers can be marked deinterleaved_ to express the fact that their
    contents is stored as a Structure of Array whenever needed.
  **/
  using deinterleaved_ = details::soa_status<true>;

  namespace tag
  {
    /// @brief Interleaved data layout option mark-up
    struct interleaving_
    {
      /// @brief Default option type
      using default_type = nt2::interleaved_;
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T>
    boost::mpl::false_ match_(interleaving_, T);

    /// INTERNAL ONLY
    template<bool B>
    boost::mpl::true_ match_(interleaving_, details::soa_status<B>);
  }
}

#endif
