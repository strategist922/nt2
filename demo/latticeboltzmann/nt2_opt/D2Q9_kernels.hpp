//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#ifndef D2Q9_KERNELS_HPP_INCLUDED
#define D2Q9_KERNELS_HPP_INCLUDED

#include <cmath>
#include <iostream>

#include <nt2/table.hpp>
#include <nt2/include/functions/of_size.hpp>
#include <nt2/include/functions/cons.hpp>

using namespace nt2;

template< typename T>
inline void relaxation( nt2::table<T> & m
                      , nt2::table<T> const s_
                      )
  {
    const T la  = T(1.);
    const T rho = T(1.);
    const T dummy_ = T(1.)/(la*la*rho);

    m(_,_,4) = m(_,_,4)*(T(1.)-s_(1))
             + s_(1)*(-T(2.)*m(_,_,1)
                     + T(3.)*( dummy_*m(_,_,2)*m(_,_,2)
                             + dummy_*m(_,_,3)*m(_,_,3)
                             )
                     );

    m(_,_,5) = m(_,_,5)*( T(1.)-s_(2))
             + s_(2)*( m(_,_,1)
                     + T(1.5)*( dummy_*m(_,_,2)*m(_,_,2)
                              + dummy_*m(_,_,3)*m(_,_,3)
                              )
                     );

    m(_,_,6) = m(_,_,6)*(T(1.)-s_(3))
             - s_(3)*m(_,_,2)/la;

    m(_,_,7) = m(_,_,7)*(T(1.)-s_(4))
             - s_(4)*m(_,_,3)/la;

    m(_,_,8) = m(_,_,8)*(T(1.)-s_(5))
             + s_(5)*( dummy_*m(_,_,2)*m(_,_,2)
                     - dummy_*m(_,_,3)*m(_,_,3)
                     );

    m(_,_,9) = m(_,_,9)*(T(1.)-s_(6))
             + s_(6)*dummy_*m(_,_,2)*m(_,_,3);
  }

template< typename T>
void get_f( nt2::table<T> const & f
           , nt2::table<T> & fcopy
           ,int nx
           ,int ny
           )
{
   for(int j=1; j<=ny; j++)
   fcopy(_,j,1) = f(_,j,1);

   for(int j=1; j<=ny; j++)
   { fcopy(1,j,2) = T(0.);
     fcopy(_(2,nx),j,2) = f(_(1,nx-1),j,2);
   }

   fcopy(_,1,3) = T(0.);
   for(int j=2; j<=ny; j++)
   fcopy(_,j,3) =  f(_,j-1,3);

   for(int j=1; j<=ny; j++)
   { fcopy(_(1,nx-1),j,4) = f(_(2,nx),j,4);
     fcopy(nx,j,4) = T(0.);
   }

   for(int j=1; j<=ny-1; j++)
   fcopy(_,j,5) = f(_,j+1,5);
   fcopy(_,ny,5) = T(0.);

   fcopy(_,1,6) = T(0.);
   for(int j=2; j<=ny; j++)
   { fcopy(1,j,6) = T(0.);
     fcopy(_(2,nx),j,6) = f(_(1,nx-1),j-1,6);
   }

   fcopy(_,1,7) = T(0.);
   for(int j=2; j<=ny; j++)
   { fcopy(_(1,nx-1),j, 7 ) = f( _(2,nx), j-1, 7 );
     fcopy(nx,j,7 ) = T(0.);
   }

   for(int j=1; j<=ny-1; j++)
   { fcopy(_(1,nx-1),j, 8 ) = f( _(2,nx),j+1, 8 );
     fcopy(nx,j,8) = T(0.);
   }
   fcopy(_,ny,8) = T(0.);

   for(int j=1; j<=ny-1; j++)
   { fcopy(1,j,9) = T(0.);
     fcopy(_(2,nx),j, 9 ) = f(_(1,nx-1),j+1, 9 );
   }
   fcopy(_,ny,9) = T(0.);
}

template<typename T>
inline void f2m( nt2::table<T> & in
               , nt2::table<T> & out
               )
{
  const T la = T(1.);
  out(_,_,1) = in(_,_,1)+in(_,_,2)+in(_,_,3)+in(_,_,4)+in(_,_,5)+in(_,_,6)+in(_,_,7)+in(_,_,8)+in(_,_,9);
  out(_,_,2) = la*(in(_,_,2)-in(_,_,4)+in(_,_,6)-in(_,_,7)-in(_,_,8)+in(_,_,9));
  out(_,_,3) = la*(in(_,_,3)-in(_,_,5)+in(_,_,6)+in(_,_,7)-in(_,_,8)-in(_,_,9));
  out(_,_,4) = -T(4.)*in(_,_,1)-in(_,_,2)-in(_,_,3)-in(_,_,4)-in(_,_,5)+T(2.)*(in(_,_,6)+in(_,_,7)+in(_,_,8)+in(_,_,9));
  out(_,_,5) = T(4.)*in(_,_,1)-T(2.)*(in(_,_,2)+in(_,_,3)+in(_,_,4)+in(_,_,5))+in(_,_,6)+in(_,_,7)+in(_,_,8)+in(_,_,9);
  out(_,_,6) = T(2.)*(-in(_,_,2)+in(_,_,4))+in(_,_,6)-in(_,_,7)-in(_,_,8)+in(_,_,9);
  out(_,_,7) = T(2.)*(-in(_,_,3)+in(_,_,5))+in(_,_,6)+in(_,_,7)-in(_,_,8)-in(_,_,9);
  out(_,_,8) = in(_,_,2)-in(_,_,3)+in(_,_,4)-in(_,_,5);
  out(_,_,9) = in(_,_,6)-in(_,_,7)+in(_,_,8)-in(_,_,9);
}

template<typename T>
inline void m2f( nt2::table<T> & in
               , nt2::table<T> & out
               )
{
  const T la = T(1.);
  const T a  = T(1./9.)
        , b  = T(1./36.)
        , c = T(1.)/(T(6.)*la)
        , d = T(1.)/T(12.)
        , e = T(.25);

  out(_,_,1) = a*in(_,_,1)-T(4.)*b*(in(_,_,4)-in(_,_,5));
  out(_,_,2) = a*in(_,_,1)+c*in(_,_,2)-b*in(_,_,4)-T(2.)*b*in(_,_,5)-T(2.)*d*in(_,6)+e*in(_,8);
  out(_,_,3) = a*in(_,_,1)+c*in(_,_,3)-b*in(_,_,4)-T(2.)*b*in(_,_,5)-T(2.)*d*in(_,7)-e*in(_,8);
  out(_,_,4) = a*in(_,_,1)-c*in(_,_,2)-b*in(_,_,4)-T(2.)*b*in(_,_,5)+T(2.)*d*in(_,6)+e*in(_,8);
  out(_,_,5) = a*in(_,_,1)-c*in(_,_,3)-b*in(_,_,4)-T(2.)*b*in(_,_,5)+T(2.)*d*in(_,7)-e*in(_,8);
  out(_,_,6) = a*in(_,_,1)+c*in(_,_,2)+c*in(_,_,3)+T(2.)*b*in(_,_,4)+b*in(_,_,5)+d*in(_,_,6)+d*in(_,_,7)+e*in(_,_,9);
  out(_,_,7) = a*in(_,_,1)-c*in(_,_,2)+c*in(_,_,3)+T(2.)*b*in(_,_,4)+b*in(_,_,5)-d*in(_,_,6)+d*in(_,_,7)-e*in(_,_,9);
  out(_,_,8) = a*in(_,_,1)-c*in(_,_,2)-c*in(_,_,3)+T(2.)*b*in(_,_,4)+b*in(_,_,5)-d*in(_,_,6)-d*in(_,_,7)+e*in(_,_,9);
  out(_,_,9) = a*in(_,_,1)+c*in(_,_,2)-c*in(_,_,3)+T(2.)*b*in(_,_,4)+b*in(_,_,5)+d*in(_,_,6)-d*in(_,_,7)-e*in(_,_,9);

}

template< typename T>
inline void bouzidi(  nt2::table<T> & f
                    , nt2::table<T> & fcopy
                    , int k
                    , nt2::table<int> & bc
                    , nt2::table<int> & alpha
                    , int nx
                    , int ny
                    )
{
  nt2::table<int,nt2::of_size_<9> > invalpha
  (nt2::cons<int>(1, 4, 5, 2, 3, 8, 9, 6, 7));

  const T q = T(.5);

  fcopy(_,_,invalpha(k))
    //bounce back conditions
  = nt2::if_else( (alpha >> (k-2)&1)
        ,nt2::if_else( (bc == 1)
             ,nt2::if_else(q*nt2::ones(nt2::of_size(nx,ny), nt2::meta::as_<T>()) < T(.5)
                  ,(T(1.) - T(2.)*q)*fcopy(_,_,k) + T(2.)*q*f(_,_,k) + fcopy(_,_,invalpha(k))
                  ,(T(1.) - T(.5)/q)*f(_,_,invalpha(k)) +T(.5)/q*f(_,_,k) + fcopy(_,_,invalpha(k))
                  )
    //anti bounce back conditions
             , nt2::if_else( (bc == 2)
                   ,nt2::if_else(q*nt2::ones(nt2::of_size(nx,ny), nt2::meta::as_<T>())< T(.5)
                        ,-(T(1.) - T(2.)*q)*fcopy(_,_,k) - T(2.)*q*f(_,_,k) + fcopy(_,_,invalpha(k))
                        ,-(T(1.) - T(.5)/q)*f(_,_,invalpha(k)) -T(.5)/q*f(_,_,k) + fcopy(_,_,invalpha(k))
                        )
    //Neumann conditions
                   ,nt2::if_else( (bc == 3)
                        , f(_,_,invalpha(k))
                        , fcopy(_,_,invalpha(k))
                        )
                    )
              )
        ,fcopy(_,_,invalpha(k))
        );
}


template< typename T>
inline void apply_bc( nt2::table<T> & f
                    , nt2::table<T> & fcopy
                    , nt2::table<int> & bc
                    , nt2::table<int> & alpha
                    , int nx
                    , int ny
                    )
{
    for(int k=2;k<=9;k++)
      bouzidi(f, fcopy, k, bc, alpha, nx, ny);
}

#endif
