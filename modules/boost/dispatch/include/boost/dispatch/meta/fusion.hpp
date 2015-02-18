//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
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
#include <boost/dispatch/meta/is_homogeneous.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/proto/traits.hpp>
#include <boost/mpl/size_t.hpp>
#include <boost/mpl/at.hpp>
#include <boost/array.hpp>
#include <array>

namespace boost { namespace dispatch { namespace meta
{
  /// @brief Fusion sequence hierarchy type
  template<typename T, typename Sz>
  struct fusion_sequence_ : unspecified_<T>
  {
    typedef unspecified_<T> parent;
  };

  /// @brief Homogeneous fusion sequence hierarchy type
  template<typename T, typename N>
  struct homogeneous_ : homogeneous_<typename T::parent, N>
  {
    typedef homogeneous_<typename T::parent, N> parent;
  };

  template<typename T, typename N>
  struct homogeneous_<unspecified_<T>, N> : fusion_sequence_<T,N>
  {
    typedef fusion_sequence_<T,N> parent;
  };

  /// @brief array hierarchy type
  template<typename T, typename N>
  struct array_ : array_<typename T::parent, N>
  {
    typedef array_<typename T::parent, N> parent;
  };

  template<typename T, typename N>
  struct  array_<unspecified_<T>, N>
        : homogeneous_<typename hierarchy_of<typename meta::scalar_of<T>::type,T>::type,N>
  {
    typedef homogeneous_<typename hierarchy_of<typename meta::scalar_of<T>::type,T>::type,N> parent;
  };

  /// INTERNAL ONLY
  template<typename T, std::size_t N>
  struct value_of< boost::array<T,N> >
  {
    typedef T type;
  };

  /// INTERNAL ONLY
  template<typename T, std::size_t N>
  struct value_of< std::array<T,N> >
  {
    typedef T type;
  };

  /// INTERNAL ONLY
  template<typename T, std::size_t N>
  struct model_of< boost::array<T, N> >
  {
    struct type
    {
      template<typename X>
      struct apply
      {
        typedef boost::array<X, N> type;
      };
    };
  };

  /// INTERNAL ONLY
  template<typename T, std::size_t N>
  struct model_of< std::array<T, N> >
  {
    struct type
    {
      template<typename X>
      struct apply
      {
        typedef std::array<X, N> type;
      };
    };
  };
}

namespace details
{
  template<typename T>
  struct is_array : boost::mpl::false_ {};

  template<typename T, std::size_t N>
  struct is_array< boost::array<T, N> > : boost::mpl::true_ {};

  template<typename T, std::size_t N>
  struct is_array< std::array<T, N> > : boost::mpl::true_ {};

  template<typename T,typename Origin>
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
    using status = typename meta::is_homogeneous<T>::type;
    using base   = typename boost::mpl::at_c<T,0>::type;
    using hbase  = typename meta::hierarchy_of<base,Origin>::type;
    using size   = boost::mpl::size_t<boost::mpl::size<T>::value>;

    using type = typename boost::mpl::if_ < status
                                          , meta::homogeneous_<hbase, size>
                                          , meta::fusion_sequence_<Origin, size>
                                          >::type;
  };

  /// Homogeneous sequence DOES have a value_of
  template<typename T>
  struct value_of < T
                  , typename boost
                    ::enable_if_c < boost::fusion
                                  ::traits::is_sequence<T>::value
                                  && !is_array<T>::value
                                  && !proto::is_expr<T>::value
                                  >::type
                  >
  {
    using status = typename meta::is_homogeneous<T>::type;
    using base   = typename boost::mpl::at_c<T,0>::type;
    using hbase  = typename meta::value_of<base>::type;
    using type = typename boost::mpl::if_<status, hbase, T>::type;
  };

  template<typename T,typename Origin>
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
    using status = typename meta::is_homogeneous<T>::type;
    using base   = typename boost::mpl::at_c<T,0>::type;
    using hbase  = typename meta::property_of<base,Origin>::type;
    using size   = boost::mpl::size_t<boost::mpl::size<T>::value>;

    using type = typename boost::mpl::if_ < status
                                          , hbase
                                          , meta::fusion_sequence_<Origin, size>
                                          >::type;
  };
}

namespace meta
{
  /// INTERNAL ONLY
  template<typename T, std::size_t N,typename Origin>
  struct  hierarchy_of< boost::array<T,N>, Origin >
  {
    typedef array_< typename hierarchy_of<T, Origin>::type
                  , boost::mpl::size_t<N>
                  > type;
  };

  /// INTERNAL ONLY
  template<typename T, std::size_t N,typename Origin>
  struct  hierarchy_of< std::array<T,N>, Origin >
  {
    typedef array_< typename hierarchy_of<T, Origin>::type
                  , boost::mpl::size_t<N>
                  > type;
  };
} } }

#endif
