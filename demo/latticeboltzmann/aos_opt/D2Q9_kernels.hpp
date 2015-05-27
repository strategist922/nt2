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
#include <nt2/include/functions/tie.hpp>

using namespace nt2;

template< typename T>
inline void relaxation( nt2::table<T> & m
                      , nt2::table<T> const s_
                      )
  {
    const T la  = T(1.);
    const T rho = T(1.);
    const T dummy_ = T(1.)/(la*la*rho);

   nt2::tie( m(4,_,_)
           , m(5,_,_)
           , m(6,_,_)
           , m(7,_,_)
           , m(8,_,_)
           , m(9,_,_)
           )
   = nt2::tie(
               m(4,_,_)*(T(1.)-s_(1))
                       + s_(1)*(-T(2.)*m(1,_,_)
                               + T(3.)*( dummy_*m(2,_,_)*m(2,_,_)
                                       + dummy_*m(3,_,_)*m(3,_,_)
                                       )
                               )

             , m(5,_,_)*( T(1.)-s_(2))
                     + s_(2)*( m(1,_,_)
                             + T(1.5)*( dummy_*m(2,_,_)*m(2,_,_)
                                      + dummy_*m(3,_,_)*m(3,_,_)
                                      )
                             )

             , m(6,_,_)*(T(1.)-s_(3))
                     - s_(3)*m(2,_,_)/la

             , m(7,_,_)*(T(1.)-s_(4))
                     - s_(4)*m(3,_,_)/la

             , m(8,_,_)*(T(1.)-s_(5))
                     + s_(5)*( dummy_*m(2,_,_)*m(2,_,_)
                             - dummy_*m(3,_,_)*m(3,_,_)
                             )

             , m(9,_,_)*(T(1.)-s_(6))
                     + s_(6)*dummy_*m(2,_,_)*m(3,_,_)
             );
  }

template< typename T>
void get_f( nt2::table<T> const & f
           , nt2::table<T> & fcopy
           ,int nx
           ,int ny
           )
{

    for(int j = 1; j<=ny; j++)
    {
      for(int i = 1; i<=nx; i++)
      {
        fcopy(1,i,j) =                 f( 1, i   , j   );
        fcopy(2,i,j) = (i>1)         ? f( 2, i-1 , j   ) : T(0.);
        fcopy(3,i,j) = (j>1)         ? f( 3, i   , j-1 ) : T(0.);
        fcopy(4,i,j) = (i<nx)        ? f( 4, i+1 , j   ) : T(0.);
        fcopy(5,i,j) = (j<ny)        ? f( 5, i   , j+1 ) : T(0.);
        fcopy(6,i,j) = (i>1 && j>1)  ? f( 6, i-1 , j-1 ) : T(0.);
        fcopy(7,i,j) = (i<nx && j>1) ? f( 7, i+1 , j-1 ) : T(0.);
        fcopy(8,i,j) = (i<nx && j<ny)? f( 8, i+1 , j+1 ) : T(0.);
        fcopy(9,i,j) = (i>1 && j<ny) ? f( 9, i-1 , j+1 ) : T(0.);
      }
    }
}

template<typename T>
inline void f2m( nt2::table<T> & in
               , nt2::table<T> & out
               )
{
  const T la = T(1.);

  nt2::tie
  ( out(1,_,_)
  , out(2,_,_)
  , out(3,_,_)
  , out(4,_,_)
  , out(5,_,_)
  , out(6,_,_)
  // , out(7,_,_)
  // , out(8,_,_)
  // , out(9,_,_)
  )
  = nt2::tie
    ( in(1,_,_)+in(2,_,_)+in(3,_,_)+in(4,_,_)+in(5,_,_)+in(6,_,_)+in(7,_,_)+in(8,_,_)+in(9,_,_)
    , la*(in(2,_,_)-in(4,_,_)+in(6,_,_)-in(7,_,_)-in(8,_,_)+in(9,_,_))
    , la*(in(3,_,_)-in(5,_,_)+in(6,_,_)+in(7,_,_)-in(8,_,_)-in(9,_,_))
    , -T(4.)*in(1,_,_)-in(2,_,_)-in(3,_,_)-in(4,_,_)-in(5,_,_)+T(2.)*(in(6,_,_)+in(7,_,_)+in(8,_,_)+in(9,_,_))
    , T(4.)*in(1,_,_)-T(2.)*(in(2,_,_)+in(3,_,_)+in(4,_,_)+in(5,_,_))+in(6,_,_)+in(7,_,_)+in(8,_,_)+in(9,_,_)
    , T(2.)*(-in(2,_,_)+in(4,_,_))+in(6,_,_)-in(7,_,_)-in(8,_,_)+in(9,_,_)
    // , T(2.)*(-in(3,_,_)+in(5,_,_))+in(6,_,_)+in(7,_,_)-in(8,_,_)-in(9,_,_)
    // , in(2,_,_)-in(3,_,_)+in(4,_,_)-in(5,_,_)
    // , in(6,_,_)-in(7,_,_)+in(8,_,_)-in(9,_,_)
    );
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

  nt2::tie
  ( out(1,_,_)
  , out(2,_,_)
  , out(3,_,_)
  , out(4,_,_)
  , out(5,_,_)
  , out(6,_,_)
  // , out(7,_,_)
  // , out(8,_,_)
  // , out(9,_,_)
  )
  = nt2::tie
     ( a*in(1,_,_)-T(4.)*b*(in(4,_,_)-in(5,_,_))
     , a*in(1,_,_)+c*in(2,_,_)-b*in(4,_,_)-T(2.)*b*in(5,_,_)-T(2.)*d*in(6,_,_)+e*in(8,_,_)
     , a*in(1,_,_)+c*in(3,_,_)-b*in(4,_,_)-T(2.)*b*in(5,_,_)-T(2.)*d*in(7,_,_)-e*in(8,_,_)
     , a*in(1,_,_)-c*in(2,_,_)-b*in(4,_,_)-T(2.)*b*in(5,_,_)+T(2.)*d*in(6,_,_)+e*in(8,_,_)
     , a*in(1,_,_)-c*in(3,_,_)-b*in(4,_,_)-T(2.)*b*in(5,_,_)+T(2.)*d*in(7,_,_)-e*in(8,_,_)
     , a*in(1,_,_)+c*in(2,_,_)+c*in(3,_,_)+T(2.)*b*in(4,_,_)+b*in(5,_,_)+d*in(6,_,_)+d*in(7,_,_)+e*in(9,_,_)
     // , a*in(1,_,_)-c*in(2,_,_)+c*in(3,_,_)+T(2.)*b*in(4,_,_)+b*in(5,_,_)-d*in(6,_,_)+d*in(7,_,_)-e*in(9,_,_)
     // , a*in(1,_,_)-c*in(2,_,_)-c*in(3,_,_)+T(2.)*b*in(4,_,_)+b*in(5,_,_)-d*in(6,_,_)-d*in(7,_,_)+e*in(9,_,_)
     // , a*in(1,_,_)+c*in(2,_,_)-c*in(3,_,_)+T(2.)*b*in(4,_,_)+b*in(5,_,_)+d*in(6,_,_)-d*in(7,_,_)-e*in(9,_,_)
     );

}

template< typename T>
inline nt2::table<T> bouzidi( nt2::table<T> & f
                            , nt2::table<T> & fcopy
                            , int k
                            , int l
                            , nt2::table<int> & bc
                            , nt2::table<int> & alpha
                            , int nx
                            , int ny
                            )
{

  const T q = T(.5);

    //bounce back conditions
  return nt2::if_else( (alpha >> (k-2)&1)
          ,nt2::if_else( (bc == 1)
               ,nt2::if_else(q*nt2::ones(nt2::of_size(1,nx,ny), nt2::meta::as_<T>()) <= T(.5)
                    ,(T(1.) - T(2.)*q)*fcopy(k,_,_) + T(2.)*q*f(k,_,_) + fcopy(l,_,_)
                    ,(T(1.) - T(.5)/q)*f(l,_,_) +T(.5)/q*f(k,_,_) + fcopy(l,_,_)
                    )
      //anti bounce back conditions
               , nt2::if_else( (bc == 2)
                     ,nt2::if_else(q*nt2::ones(nt2::of_size(1,nx,ny), nt2::meta::as_<T>())< T(.5)
                          ,-(T(1.) - T(2.)*q)*fcopy(k,_,_) - T(2.)*q*f(k,_,_) + fcopy(l,_,_)
                          ,-(T(1.) - T(.5)/q)*f(l,_,_) -T(.5)/q*f(k,_,_) + fcopy(l,_,_)
                          )
      //Neumann conditions
                     ,nt2::if_else( (bc == 3)
                          , f(l,_,_)
                          , fcopy(l,_,_)
                          )
                      )
                )
          ,fcopy(l,_,_)
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
    nt2::tie( fcopy(4,_,_)
            , fcopy(5,_,_)
            , fcopy(2,_,_)
            , fcopy(3,_,_)
            , fcopy(8,_,_)
            , fcopy(9,_,_)
            //, fcopy(6,_,_)
            // , fcopy(7,_,_)
            )
    = nt2::tie( bouzidi(f, fcopy, 2, 4, bc, alpha, nx, ny)
              , bouzidi(f, fcopy, 3, 5, bc, alpha, nx, ny)
              , bouzidi(f, fcopy, 4, 2, bc, alpha, nx, ny)
              , bouzidi(f, fcopy, 5, 3, bc, alpha, nx, ny)
              , bouzidi(f, fcopy, 6, 8, bc, alpha, nx, ny)
              , bouzidi(f, fcopy, 7, 9, bc, alpha, nx, ny)
              //, bouzidi(f, fcopy, 8, 6, bc, alpha, nx, ny)
              // , bouzidi(f, fcopy, 9, 7, bc, alpha, nx, ny)
              );
}

#endif
