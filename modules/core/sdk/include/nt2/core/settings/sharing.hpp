//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_SHARING_HPP_INCLUDED
#define NT2_CORE_SETTINGS_SHARING_HPP_INCLUDED

#include <boost/mpl/bool.hpp>

namespace nt2
{
  namespace details
  {
    template<bool B> struct sharing_status
    {
      using sharing_type = boost::mpl::bool_<B>;
    };
  }

  /*!
    @brief Memory ownership tag representing shared memory

    This tag indicates that current Container shares its memory with an
    external source to which it delegates the memory handling (including clean
    up of said memory).
  **/
  using shared_ = details::sharing_status<true>;

  /*!
    @brief Memory ownership tag representing owned memory

    This tag indicates that current Container owns its own memory and
    handles it on its own, including clean-up of said memory.
  **/
  using owned_  = details::sharing_status<false>;

  namespace tag
  {
    /// @brief Data ownership option mark-up
    struct sharing_
    {
      /// @brief Default option type
      using default_type = nt2::owned_;
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T>
    boost::mpl::false_ match_(sharing_, T);

    /// INTERNAL ONLY
    template<bool B>
    boost::mpl::true_ match_(sharing_, details::sharing_status<B>);
  }
}

#endif
