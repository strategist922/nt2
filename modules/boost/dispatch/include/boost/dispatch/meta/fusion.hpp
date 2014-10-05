//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_FUSION_HPP_INCLUDED
#define BOOST_DISPATCH_META_FUSION_HPP_INCLUDED

#include <boost/dispatch/meta/factory_of.hpp>
#include <boost/dispatch/meta/hierarchy_of.hpp>
#include <boost/dispatch/meta/property_of.hpp>
#include <boost/dispatch/meta/primitive_of.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/proto/traits.hpp>
#include <boost/mpl/size_t.hpp>
#include <boost/array.hpp>

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief Fusion sequence hierarchy

    Represents any Fusion Sequence of type @c Type.

    @par Model:
    Hierarchy

    @tparam Type FusionSequence to hierarchize
  **/
  template<typename Type>
  struct  fusion_sequence_
#if !defined(DOXYGEN_ONLY)
        : unspecified_<Type>
#endif
  {
    /// @brief Parent hierarchy
    typedef unspecified_<Type> parent;
  };

  /*!
    @brief Array hierarchy

    Represents an Array by using its @c Type and @c Size.

    Parent hierarchy is computed by using the parent hierarchy of @c Type.
    Once array_<unspecified_<T>,Size> is reached, the parent hierarchy becomes
    fusion_sequence_<T>.

    @par Model:
    Hierarchy

    @tparam Type Type of the hierarchized array
    @tparam Size Size of the hierarchized array
  **/
  template<typename Type, typename Size>
  struct array_
#if !defined(DOXYGEN_ONLY)
        : array_<typename Type::parent, Size>
#endif
  {
    /// @brief Parent hierarchy
    typedef array_<typename Type::parent, Size> parent;
  };

#if !defined(DOXYGEN_ONLY)
  template<typename Type, typename N>
  struct array_<unspecified_<Type>, N> : fusion_sequence_<Type>
  {
    typedef fusion_sequence_<Type> parent;
  };
#endif

  /// INTERNAL ONLY
  template<class T, std::size_t N>
  struct value_of< boost::array<T,N> >
  {
    typedef T type;
  };

  /// INTERNAL ONLY
  template<class T, std::size_t N>
  struct model_of< boost::array<T, N> >
  {
    struct type
    {
      template<class X>
      struct apply
      {
        typedef boost::array<X, N> type;
      };
    };
  };
}

namespace details
{
  template<class T>
  struct is_array : boost::mpl::false_ {};

  template<class T, std::size_t N>
  struct is_array< boost::array<T, N> > : boost::mpl::true_ {};

  template<class T,class Origin>
  struct  hierarchy_of< T
                      , Origin
                      , typename boost
                        ::enable_if_c < boost::fusion
                                        ::traits::is_sequence<T>::value
                                        && !is_array<T>::value
                                        && !proto::is_expr<T>::value
                                      >::type
                      >
  {
    typedef meta::fusion_sequence_<Origin> type;
  };

  template<class T,class Origin>
  struct   property_of< T
                      , Origin
                      , typename boost
                        ::enable_if_c < boost::fusion
                                        ::traits::is_sequence<T>::value
                                        && !is_array<T>::value
                                        && !proto::is_expr<T>::value
                                      >::type
                      >
  {
    typedef meta::fusion_sequence_<Origin> type;
  };
}

namespace meta
{
  /// INTERNAL ONLY
  template<class T, std::size_t N,class Origin>
  struct  hierarchy_of< boost::array<T,N>
                      , Origin
                      >
  {
    typedef array_< typename hierarchy_of<T, Origin>::type
                  , boost::mpl::size_t<N>
                  > type;
  };
} } }

#endif
