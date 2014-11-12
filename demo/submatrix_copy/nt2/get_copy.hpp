//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#ifndef GET_COPY_HPP_INCLUDED
#define GET_COPY_HPP_INCLUDED

#include <iostream>
#include <nt2/table.hpp>

using namespace nt2;

template< typename T>
inline void get_f( nt2::table<T> const & f
                 , nt2::table<T> & fcopy
                 , int nx
                 , int ny
                 )
{
      // fcopy(_,_,1) = f(_,_,1);
      // fcopy(_(2,nx),_,2)    = f(_(1,nx-1),_,2);
      // fcopy(_,_(2,ny),3)    = f(_,_(1,ny-1),3);
      // fcopy(_(1,nx-1),_, 4) = f(_(2,nx),_,4);
      // fcopy(_,_(1,ny-1), 5 ) = f(_,_(2,ny), 5 );
      // fcopy(_(2,nx),_(1,ny-1), 6 ) = f(_(1,nx-1),_(2,ny), 6 );
      // fcopy(_(1,nx-1),_(2,ny), 7 ) = f( _(2,nx), _(1,ny-1), 7 );
      // fcopy(_(1,nx-1), _(1,ny-1), 8 ) = f( _(2,nx),_(2,ny), 8 );
      // fcopy(_(2,nx),_(1,ny-1), 9 ) = f(_(1,nx-1),_(2,ny), 9 );

    for(std::size_t j=1; j<=ny; j++)
    for(std::size_t i=1; i<=nx; i++)
    {
      fcopy(i,j,1) = f(i,j,1);
      fcopy(i,j,2) = (i>1) ? f(i-1,j,2) : T(0.);
      fcopy(i,j,3) = (j>1) ? f(i,j-1,3) : T(0.);
      fcopy(i,j,4) = (i<nx) ? f(i+1,j,4) : T(0.);
      fcopy(i,j,5) = (j<ny)? f(i,j+1,5) : T(0.);
      fcopy(i,j,6) = (i>1 && j<ny) ? f(i-1,j+1,6) : T(0.);
      fcopy(i,j,7) = (i<nx && j>1) ? f(i+1,j-1,7) : T(0.);
      fcopy(i,j,8) = (i<nx && j<ny) ? f(i+1,j+1,8): T(0.);
      fcopy(i,j,9) = (i>1 && j<ny) ? f(i-1,j+1,9) : T(0.);
    }

}

#endif
