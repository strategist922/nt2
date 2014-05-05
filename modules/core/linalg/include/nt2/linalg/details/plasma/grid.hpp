//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_GRID_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_GRID_HPP_INCLUDED

#include <vector>

namespace nt2 {

  namespace details
  {

    template <class T>
    class Grid
    {
      public:

      std::size_t height;
      std::size_t width;
      std::vector<T> data;

      Grid(std::size_t h=0, std::size_t w=0, T const & c=T())
      :height(h),width(w),data(w*h,c)
      {}

      Grid(size_t h, size_t w, std::vector<T> &d)
      :height(h),width(w),data(d)
      {}

      ~Grid() {}

      T& operator()(std::size_t i, std::size_t j)
      { return data[i+j*height]; }

      T const & operator()(std::size_t i, std::size_t j)const
      { return data[i+j*height]; }

    };

  }
}

#endif
