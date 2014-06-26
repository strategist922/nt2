//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_DETAILS_SHAPE_HPP_INCLUDED
#define NT2_CORE_SETTINGS_DETAILS_SHAPE_HPP_INCLUDED

#include <nt2/core/settings/option.hpp>
#include <nt2/core/settings/buffer.hpp>
#include <nt2/core/functions/scalar/numel.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

#include <algorithm>

namespace nt2
{
  namespace details
  {
    template<std::ptrdiff_t Bound>
    struct upper_band
    {
      typedef boost::mpl::int<Bound> upper_bound;
      typedef boost::mpl::bool_<Bound == -1> has_open_upper_bound;
    };

    template<std::ptrdiff_t Bound>
    struct lower_band
    {
      typedef boost::mpl::int<Bound> lower_bound;
      typedef boost::mpl::bool_<Bound == -1> has_open_lowper_bound;
    };
  }

  /*!
    @brief band_diagonal_ shape  option

    @tparam UpperBound  Number of upper diagonal being non-trivial. UpperBound
                        is equal to -1 if all upper diagonals are non-trivial.

    @tparam LowerBound  Number of lower diagonal being non-trivial. LowerBound
                        is equal to -1 if all lower diagonals are non-trivial.
  **/
  template< std::ptrdiff_t UpperBound
          , std::ptrdiff_t LowerBound
          >
  struct  band_diagonal_
        : public details::upper_bound<UpperBound>
        , public details::lower_band<LowerBound>
  {
    template<typename Container> struct apply
    {
      typedef typename meta::option<Container, tag::buffer_>::type  buffer_t;
      typedef typename details::make_buffer<buffer_t>
                              ::template apply<Container>::type type;
    };

    template<typename Extent>
    static BOOST_FORCEINLINE std::size_t nnz(Extent const& sz)
    {
      return
    }
  };

  struct  band_diagonal_<0,0>
        : public details::upper_bound<0>
        , public details::lower_band<0>
  {
    template<typename Container> struct apply
    {
      typedef typename meta::option<Container, tag::buffer_>::type  buffer_t;
      typedef typename details::make_buffer<buffer_t>
                              ::template apply<Container>::type type;
    };

    template<typename Extent>
    static BOOST_FORCEINLINE std::size_t nnz(Extent const& sz)
    {
      return std::min ( boost::fusion::at_c<0>(sz)
                      , boost::fusion::at_c<1>(sz)
                      );
    }
  };
}

#endif
