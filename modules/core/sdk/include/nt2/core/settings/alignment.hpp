//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_ALIGNMENT_HPP_INCLUDED
#define NT2_CORE_SETTINGS_ALIGNMENT_HPP_INCLUDED

#include <boost/mpl/bool.hpp>

namespace nt2
{
  namespace details
  {
    template<bool B> struct aligned_status
    {
      using alignment_type = boost::mpl::bool_<B>;
    };
  }

  /*!
    @brief aligned_ option

    Containers can be marked aligned_ to express the fact that their
    data are always stored onto an aligned memory address compatible with
    SIMD processing.
  **/
  using aligned_ = details::aligned_status<true>;

  /*!
    @brief unaligned_ option

    Containers can be marked unaligned_ to express the fact that their
    data are never stored onto an aligned memory address compatible with
    SIMD processing.
  **/
  using unaligned_ = details::aligned_status<false>;

  namespace tag
  {
    /// @brief Alignment option mark-up
    struct alignment_
    {
      /// @brief Default option type
      using default_type = nt2::aligned_;
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T>
    boost::mpl::false_ match_(alignment_, T);

    /// INTERNAL ONLY
    template<bool B>
    boost::mpl::true_ match_(alignment_, details::aligned_status<B>);
  }
}

#endif
