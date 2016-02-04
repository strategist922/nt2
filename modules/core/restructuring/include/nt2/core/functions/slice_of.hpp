//==============================================================================
//             Copyright 2016 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SLICE_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SLICE_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/core/utility/share.hpp>
#include <nt2/core/container/table/table.hpp>
#include <nt2/include/functions/size.hpp>

namespace nt2
{
  /*!
    @brief Contiguous slicing

    Return the a view of a contiguous slice of a given table.
  **/
  template<typename T, typename S>
  BOOST_FORCEINLINE container::table<T,nt2::settings(nt2::shared_,S)>
  slice_of(container::table<T,S>& t, int i)
  {
    auto slice_size = nt2::size(t,1);
    container::table<T,nt2::settings(nt2::shared_,S)> v( nt2::of_size(slice_size)
                                            , share ( t.data()+(i-1)*slice_size
                                                    , t.data()+i*slice_size
                                                    )
                                            );
    return v;
  }

  /// @overload
  template<typename T, typename S>
  BOOST_FORCEINLINE  container::table<T,nt2::settings(nt2::shared_,S)>
  slice_of(container::table<T,S>& t, int i, int j)
  {
    auto slice_size = nt2::size(t,1);
    auto row_size   = nt2::size(t,2);
    auto n = slice_size*( (i-1) + row_size*(j-1));

    container::table<T,nt2::settings(nt2::shared_,S)> v( nt2::of_size(slice_size)
                                            , share ( t.data()+n
                                                    , t.data()+n+slice_size
                                                    )
                                            );
    return v;
  }

  /// @overload
  template<typename T, typename S>
  BOOST_FORCEINLINE container::table<T,nt2::settings(nt2::shared_,S)>
  slice_of(container::table<T,S>& t, int i, int j, int k)
  {
    auto slice_size = nt2::size(t,1);
    auto row_size   = nt2::size(t,2);
    auto block_size = nt2::size(t,3);
    auto n = slice_size*( (i-1) + row_size*((j-1) + block_size*(k-1)));

    container::table<T,nt2::settings(nt2::shared_,S)> v( nt2::of_size(slice_size)
                                                        , share ( t.data()+n
                                                                , t.data()+n+slice_size
                                                                )
                                                        );
    return v;
  }
}

#endif
