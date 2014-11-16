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

using nt2::tag::table_;

template< typename T>
inline void relaxation( nt2::table<T> & m
                      , nt2::table<T> const & s)
{
      T la = T(1.);
      T rhoo = T(1.);
      T dummy = T(1.)/(la*la*rhoo);

      m(4) = m(4)*(T(1.)-s(1)) + s(1)*(-T(2.)*m(1) + T(3.)*(dummy*m(2)*m(2)+dummy*m(3)*m(3)));
      m(5) = m(5)*(T(1.)-s(2)) + s(2)*(m(1) + T(1.5)*(dummy*m(2)*m(2)+dummy*m(3)*m(3)));
      m(6) = m(6)*(T(1.)-s(3)) - s(3)*m(2)/la;
      m(7) = m(7)*(T(1.)-s(4)) - s(4)*m(3)/la;
      m(8) = m(8)*(T(1.)-s(5)) + s(5)*(dummy*m(2)*m(2)-dummy*m(3)*m(3));
      m(9) = m(9)*(T(1.)-s(6)) + s(6)*dummy*m(2)*m(3);
}

template< typename T>
inline void get_f( nt2::table<T> const & f
                 , nt2::table<T> & f_loc
                 , int nx
                 , int ny
                 , int i
                 , int j
          )
{
    f_loc(1) = f( i   , j   , 1 );
    f_loc(2) = (i>1)  ? f( i-1 , j   , 2 ) : T(0.);
    f_loc(3) = (j>1)  ? f( i   , j-1 , 3 ) : T(0.);
    f_loc(4) = (i<nx) ? f( i+1 , j   , 4 ) : T(0.);
    f_loc(5) = (j<ny) ? f( i   , j+1 , 5 ) : T(0.);
    f_loc(6) = (i>1 && j>1) ? f( i-1 , j-1 , 6 ) : T(0.);
    f_loc(7) = (i<nx && j>1) ? f( i+1 , j-1 , 7 ) : T(0.);
    f_loc(8) = (i<nx && j<ny)? f( i+1 , j+1 , 8 ) : T(0.);
    f_loc(9) = (i>1 && j<ny) ? f( i-1 , j+1 , 9 ) : T(0.);
}

template<typename T>
inline void f2m( nt2::table<T> & in
               , nt2::table<T> & out)
{
   T la = T(1.);
   nt2::table<T> invF (   nt2::cons<T>( nt2::of_size(9 ,9),
                          1,  1,  1,  1,  1,  1,  1,  1,  1,
                          0, la,  0,-la,  0, la,-la,-la, la,
                          0,  0, la,  0,-la, la, la,-la,-la,
                         -4, -1, -1, -1, -1,  2,  2,  2,  2,
                          4, -2, -2, -2, -2,  1,  1,  1,  1,
                          0, -2,  0,  2,  0,  1, -1, -1,  1,
                          0,  0, -2,  0,  2,  1,  1, -1, -1,
                          0,  1, -1,  1, -1,  0,  0,  0,  0,
                          0,  0,  0,  0,  0,  1, -1,  1, -1
                          )
                       );

   out.resize(nt2::of_size(1,9));
   in.resize(nt2::of_size(1,9));

   out = nt2::mtimes(in,invF);

   out.resize(nt2::of_size(9));
   in.resize(nt2::of_size(9));
}

template<typename T>
inline void m2f( nt2::table<T> & in
               , nt2::table<T> & out)
{
  T la = T(1.);
  T a  = T(1./9.)
  , b  = T(1./36.)
  , c = T(1.)/(T(6.)*la)
  , d = T(1.)/T(12.)
  , e = T(.25);

   nt2::table<T> invM (   nt2::cons<T>( nt2::of_size(9 ,9),
                          a,  0,  0, -4*b,  4*b,    0,    0,  0,  0,
                          a,  c,  0,   -b, -2*b, -2*d,    0,  e,  0,
                          a,  0,  c,   -b, -2*b,    0, -2*d, -e,  0,
                          a, -c,  0,   -b, -2*b,  2*d,    0,  e,  0,
                          a,  0, -c,   -b, -2*b,    0,  2*d, -e,  0,
                          a,  c,  c,  2*b,    b,    d,    d,  0,  e,
                          a, -c,  c,  2*b,    b,   -d,    d,  0, -e,
                          a, -c, -c,  2*b,    b,   -d,   -d,  0,  e,
                          a,  c, -c,  2*b,    b,    d,   -d,  0, -e
                          )
                       );

   out.resize(nt2::of_size(1,9));
   in.resize(nt2::of_size(1,9));

   out = nt2::mtimes(in,invM);

   out.resize(nt2::of_size(9));
   in.resize(nt2::of_size(9));

}

template< typename T>
inline void set_f( nt2::table<T> & f
                 , nt2::table<T> const & f_loc
                 , int i
                 , int j
                 )
{
  for(int k=1;k<=9;k++)
      f(i,j,k) = f_loc(k);
}

template< typename T>
inline void bouzidi( nt2::table<T> const & f
            , nt2::table<T> & f_loc
            , T rhs
            , int alpha
            , int type
            , int i
            , int j
            )
{
    nt2::table<int> invalpha
    (nt2::cons<int>(1, 4, 5, 2, 3, 8, 9, 6, 7));

    T f1, f2, q;

    rhs = f_loc(invalpha(alpha));
    q = T(.5);

    f1 = f(i,j,alpha);
    f2 = f(i,j,invalpha(alpha));

    //bounce back conditions
    if (type == 1)
    {
        if (q<=T(.5))
            f_loc(invalpha(alpha)) = (T(1.) - T(2.)*q)*f_loc(alpha) + T(2.)*q*f1 + rhs;
        else
            f_loc(invalpha(alpha)) = (T(1.) - T(.5)/q)*f2 +T(.5)/q*f1 + rhs;
    }
    //anti bounce back conditions
    else if (type == 2)
    {
        if (q<T(.5))
            f_loc(invalpha(alpha)) = -(T(1.) - T(2.)*q)*f_loc(alpha) - T(2.)*q*f1 + rhs;
        else
            f_loc(invalpha(alpha)) = -(T(1.) - T(.5)/q)*f2 -T(.5)/q*f1 + rhs;
    }
    //Neumann conditions
    else if (type == 3)
    {
        f_loc(invalpha(alpha)) = f2;
    }
}


template< typename T>
inline void apply_bc( nt2::table<T> const & f
                    , nt2::table<T> & f_loc
                    , int bc
                    , nt2::table<int> const & alphaTab
                    , int i
                    , int j
             )
{
    int alpha = alphaTab(i,j);

    for(int k=0;k<8;k++){
        if (alpha>>k&1){
            bouzidi(f, f_loc, T(0.), k+2, bc, i, j);
        }
    }
}

template< typename T>
inline void onetime_step(  nt2::table<T> & f
                   ,nt2::table<T> & fcopy
                   ,nt2::table<int> & bc
                   ,nt2::table<int> & alpha
                   ,nt2::table<T> const & s
                   ,int i
                   ,int j
                   ,int nx
                   ,int ny
                  )
{
    nt2::table<T> m_loc( nt2::of_size(9) );
    nt2::table<T> f_loc( nt2::of_size(9) );

    int bc_ = bc(i,j);

    get_f(f, f_loc, nx, ny, i, j);
    apply_bc(f, f_loc, bc_, alpha, i, j);
    f2m(f_loc, m_loc);
    relaxation(m_loc,s);
    m2f(m_loc, f_loc);
    set_f(fcopy, f_loc, i, j);

}

#endif
