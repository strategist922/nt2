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
    fcopy(_,_,1) = f(_,_,1);
    fcopy(_(2,nx),_,2)    = f(_(1,nx-1),_,2);
    fcopy(_,_(2,ny),3)    = f(_,_(1,ny-1),3);
    fcopy(_(1,nx-1),_, 4) = f(_(2,nx),_,4);
    fcopy(_,_(1,ny-1), 5 ) = f(_,_(2,ny), 5 );
    fcopy(_(2,nx),_(1,ny-1), 6 ) = f(_(1,nx-1),_(2,ny), 6 );
    fcopy(_(1,nx-1),_(2,ny), 7 ) = f( _(2,nx), _(1,ny-1), 7 );
    fcopy(_(1,nx-1), _(1,ny-1), 8 ) = f( _(2,nx),_(2,ny), 8 );
    fcopy(_(2,nx),_(1,ny-1), 9 ) = f(_(1,nx-1),_(2,ny), 9 );
}

#endif
