//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_INDEX_HPP_INCLUDED
#define NT2_CORE_SETTINGS_INDEX_HPP_INCLUDED

#include <cstddef>
#include <type_traits>

namespace nt2
{
  /*!
    @brief General base index option

    Containers can be marked using index_ to express the fact that their
    base index for the Nth dimension is given by the Nth integral constant
    contained in index_. For dimensions above the one listed, the base index
    used is the lat valid one.

    @tparam Is List of integral constant decribing the list of base indices
  **/
  template<std::ptrdiff_t... Is> struct index_
  {
    /// INTERNAL ONLY
    template<std::size_t N, bool IsValid = (N<sizeof...(Is)) >
    struct impl
    {
      template<std::ptrdiff_t V0, std::ptrdiff_t... Vs> struct at
      {
        using type = typename impl<N-1>::template at<Vs...>::type;
      };
    };

    /// INTERNAL ONLY
    template<bool IsValid> struct impl<0,IsValid>
    {
      template<std::ptrdiff_t V0, std::ptrdiff_t... Vs> struct at
      {
        using type = std::integral_constant<std::ptrdiff_t,V0>;
      };
    };

    // Handle out of range access by returning the last dimension base index
    /// INTERNAL ONLY
    template<std::size_t N> struct impl<N,false>
    {
      template<std::ptrdiff_t... Vs> struct at
      {
        using last = impl<sizeof...(Is)-1>;
        using type = typename last::template at<Vs...>::type;
      };
    };

    // MSVC need this trampoline to not ICE on using at = ...
    /// INTERNAL ONLY
    template<std::size_t N> struct at_impl
    {
      using type = typename impl<N>::template at<Is...>::type;
    };

    /// @brief Compile-time access to the Nth base index
    template<std::size_t N> using at = typename at_impl<N>::type;

    /// @brief Compile-time access to the base indices option
    using index_type = index_<Is...>;
  };

  /*!
    @brief C_index_ option

    Containers can be marked C_index_ to express the fact that their
    base index follows the C language rule by being equal to 0 on all
    dimensions.
  **/
  using C_index_  = index_<0>;

  /*!
    @brief matlab_index_ option

    Containers can be marked matlab_index_ to express the fact that their
    base index follows the Matlab language rule by being equal to 1 on all
    dimensions.
  **/
  using matlab_index_ = index_<1>;

  namespace tag
  {
    /// @brief Base indices option mark-up
    struct index_
    {
      /// @brief Default option type
      using default_type = nt2::matlab_index_;
    };

    //--------------------------------------------------------------------------
    /// INTERNAL ONLY
    template<typename T> boost::mpl::false_ match_(index_, T);

    /// INTERNAL ONLY
    template<std::ptrdiff_t... Is>
    boost::mpl::true_ match_(index_, nt2::index_<Is...>);
  }
}

#endif
