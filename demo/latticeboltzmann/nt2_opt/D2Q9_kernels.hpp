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
#include <array>

#include <nt2/table.hpp>
#include <nt2/include/functions/of_size.hpp>
#include <nt2/include/functions/cons.hpp>

#include <nt2/linalg/details/blas/mm.hpp>

using namespace nt2;

template< typename T>
inline void relaxation( nt2::table<T> & m
                      , nt2::table<T,nt2::of_size_<6> > const s_
                      )
  {
    T la  = T(1.);
    T rho = T(1.);
    T dummy_ = T(1.)/(la*la*rho);

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
inline void get_f( nt2::table<T> const & f
                 , nt2::table<T> & fcopy
                 ,int nx
                 ,int ny
                 )
{
   // fcopy(_,_,1) = f(_,_,1);
   // fcopy(_(2,nx),_,2)    = f(_(1,nx-1),_,2);
   // fcopy(_,_(2,ny),3)    = f(_,_(1,ny-1),3);
   // fcopy(_(1,nx-1),_, 4) = f(_(2,nx),_,4);
   // fcopy(_,_(1,ny-1), 5 ) = f(_,_(2,ny), 5 );
   // fcopy(_(2,nx),_(2,ny), 6 ) = f(_(1,nx-1),_(1,ny-1), 6 );
   // fcopy(_(1,nx-1),_(2,ny), 7 ) = f( _(2,nx), _(1,ny-1), 7 );
   // fcopy(_(1,nx-1), _(1,ny-1), 8 ) = f( _(2,nx),_(2,ny), 8 );
   // fcopy(_(2,nx),_(1,ny-1), 9 ) = f(_(1,nx-1),_(2,ny), 9 );

   for(int j=1; j<=ny; j++)
   for(int i=1; i<=nx; i++)
   {
     fcopy(i,j,1) = f(i,j,1);
     fcopy(i,j,2) = (i>1) ? f(i-1,j,2) : T(0.);
     fcopy(i,j,3) = (j>1) ? f(i,j-1,3) : T(0.);
     fcopy(i,j,4) = (i<nx) ? f(i+1,j,4) : T(0.);
     fcopy(i,j,5) = (j<ny)? f(i,j+1,5) : T(0.);
     fcopy(i,j,6) = (i>1 && j>1) ? f(i-1,j-1,6) : T(0.);
     fcopy(i,j,7) = (i<nx && j>1) ? f(i+1,j-1,7) : T(0.);
     fcopy(i,j,8) = (i<nx && j<ny) ? f(i+1,j+1,8): T(0.);
     fcopy(i,j,9) = (i>1 && j<ny) ? f(i-1,j+1,9) : T(0.);
   }
}

template<typename T>
inline void f2m( nt2::table<T> & fcopy
               , nt2::table<T> & m
               , int nx
               , int ny
               )
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

   fcopy.resize(nt2::of_size(nx*ny,9));
   m.resize(nt2::of_size(nx*ny,9));

   m = nt2::mtimes(fcopy,invF);

   fcopy.resize(nt2::of_size(nx,ny,9));
   m.resize(nt2::of_size(nx,ny,9));
}

template<typename T>
inline void m2f( nt2::table<T> & m
               , nt2::table<T> & f
               , int nx
               , int ny
               )
{
    T la = T(1.);
    T a(1./9)
    , b(1./36.)
    , c(1./(6.*la))
    , d(1./12)
    , e(1./4.);

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

   m.resize(nt2::of_size(nx*ny,9));
   f.resize(nt2::of_size(nx*ny,9));

   f = nt2::mtimes(m,invM);

   m.resize(nt2::of_size(nx,ny,9));
   f.resize(nt2::of_size(nx,ny,9));
}

template< typename T>
inline void bouzidi( nt2::table<T> const & f
            , nt2::table<T> & fcopy
            , int k
            , nt2::table<int> const & bc
            , nt2::table<int> const & alpha
            , int nx
            , int ny
            )
{
  nt2::table<int,nt2::of_size_<9> > invalpha
  (nt2::cons<int>(1, 4, 5, 2, 3, 8, 9, 6, 7));

  T q = T(.5);

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
inline void apply_bc( nt2::table<T> const & f
                    , nt2::table<T> & fcopy
                    , nt2::table<int> const & bc
                    , nt2::table<int> const & alpha
                    , int nx
                    , int ny
                    )
{
    for(int k=2;k<=9;k++)
      bouzidi(f, fcopy, k, bc, alpha, nx, ny);
}

template< typename T>
inline void onetime_step( nt2::table<T> & f
                        , nt2::table<T> & fcopy
                        , nt2::table<T> & m
                        , nt2::table<int> & bc
                        , nt2::table<int> & alpha
                        , nt2::table< T, nt2::of_size_<6> > const & s
                        , int nx
                        , int ny
                        )
{
  get_f(f,fcopy,nx,ny);
  apply_bc(f, fcopy, bc, alpha, nx, ny);
  f2m(fcopy, m, nx, ny);
  relaxation(m,s);
  m2f(m, fcopy, nx, ny);
}

/********************************************************************************************/
template< typename T>
inline void relaxation_scalar(std::array<T,9> & m, std::array<T,6> const & s)
{
      T la = T(1.);
      T rhoo = T(1.);
      T dummy = T(1.)/(la*la*rhoo);
      m[3] = m[3]*(T(1.)-s[0]) + s[0]*(-T(2.)*m[0] + T(3.)*(dummy*m[1]*m[1]+dummy*m[2]*m[2]));
      m[4] = m[4]*(T(1.)-s[1]) + s[1]*(m[0] + T(1.5)*(dummy*m[1]*m[1]+dummy*m[2]*m[2]));
      m[5] = m[5]*(T(1.)-s[2]) - s[2]*m[1]/la;
      m[6] = m[6]*(T(1.)-s[3]) - s[3]*m[2]/la;
      m[7] = m[7]*(T(1.)-s[4]) + s[4]*(dummy*m[1]*m[1]-dummy*m[2]*m[2]);
      m[8] = m[8]*(T(1.)-s[5]) + s[5]*dummy*m[1]*m[2];
}

template< typename T>
void get_f_scalar( std::vector<T> const & f
          , std::array<T,9> & f_loc
          , int nx
          , int ny
          , int i
          , int j
          )
{
    int dec = nx*ny;
    int ind = 0;

    f_loc[0] = f[i + j*nx + ind];
    ind += dec;

    f_loc[1] = (i>0) ? f[(i-1) + j*nx + ind] : T(0);
    ind += dec;

    f_loc[2] = (j>0) ? f[i + (j-1)*nx + ind] : T(0);
    ind += dec;

    f_loc[3] = (i<nx-1) ? f[(i+1) + j*nx + ind] : T(0);
    ind += dec;

    f_loc[4] = (j<ny-1) ? f[i + (j+1)*nx + ind] : T(0);
    ind += dec;

    f_loc[5] = ((i>0) && (j>0)) ? f[(i-1) + (j-1)*nx + ind] : T(0);
    ind += dec;

    f_loc[6] = ((i<nx-1) && (j>0)) ? f[(i+1) + (j-1)*nx + ind] : T(0);
    ind += dec;

    f_loc[7] = ((i<nx-1) && (j<ny-1)) ? f[(i+1) + (j+1)*nx + ind] : T(0);
    ind += dec;

    f_loc[8] = ((i>0) && (j<ny-1)) ? f[(i-1) + (j+1)*nx + ind] : T(0);
}

template<typename T>
inline void f2m_scalar(std::array<T,9> const & in, std::array<T,9> & out)
{
  T la   = T(1.);
  T one  = T(1.);
  T zero = T(0.);

  int inc    = 1;
  int nine   = 9;

  std::vector<T> invF
  =  {
      1,  1,  1,  1,  1,  1,  1,  1,  1,
      0, la,  0,-la,  0, la,-la,-la, la,
      0,  0, la,  0,-la, la, la,-la,-la,
     -4, -1, -1, -1, -1,  2,  2,  2,  2,
      4, -2, -2, -2, -2,  1,  1,  1,  1,
      0, -2,  0,  2,  0,  1, -1, -1,  1,
      0,  0, -2,  0,  2,  1,  1, -1, -1,
      0,  1, -1,  1, -1,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  1, -1,  1, -1
     };

// Row Major Matrix-Matrix multiplication with Column Major Blas
  nt2::details::
  gemm( "N", "N"
    , &inc, &nine, &nine
    , &one
    , & in[0], &inc
    , & invF[0], &nine
    , &zero
    , &out[0], &inc
    );
}

template<typename T>
inline void m2f_scalar(std::array<T,9> const & in, std::array<T,9> & out)
{
    T la = T(1.);
    T a  = T(1./9.)
    , b  = T(1./36.)
    , c = T(1.)/(T(6.)*la)
    , d = T(1.)/T(12.)
    , e = T(.25);

    T one  = T(1.);
    T zero = T(0.);

    int inc    = 1;
    int nine   = 9;

    std::vector<T> invM
    =  {
          a,  0,  0, -4*b,  4*b,    0,    0,  0,  0,
          a,  c,  0,   -b, -2*b, -2*d,    0,  e,  0,
          a,  0,  c,   -b, -2*b,    0, -2*d, -e,  0,
          a, -c,  0,   -b, -2*b,  2*d,    0,  e,  0,
          a,  0, -c,   -b, -2*b,    0,  2*d, -e,  0,
          a,  c,  c,  2*b,    b,    d,    d,  0,  e,
          a, -c,  c,  2*b,    b,   -d,    d,  0, -e,
          a, -c, -c,  2*b,    b,   -d,   -d,  0,  e,
          a,  c, -c,  2*b,    b,    d,   -d,  0, -e
       };

    nt2::details::
    gemm( "N", "N"
      , &inc, &nine, &nine
      , &one
      , & in[0], &inc
      , & invM[0], &nine
      , &zero
      , &out[0], &inc
      );
}

template< typename T>
inline void set_f_scalar( std::vector<T> & f
          , std::array<T,9> const & f_loc
          , int nx
          , int ny
          , int i
          , int j
          )
{
  int dec = nx*ny;
  int ind = i +j*nx;

  for(int k=0;k<9;k++){
      f[ind] = f_loc[k];
      ind += dec;
  }
}

template< typename T>
inline void bouzidi_scalar( std::vector<T> const & f
            , std::array<T,9> & f_loc
            , T rhs
            , int alpha
            , int type
            , int nx
            , int ny
            , int i
            , int j
            )
{
    int dec = nx*ny;
    std::array<int,9> invalpha={0, 3, 4, 1, 2, 7, 8, 5, 6};
    T f1, f2, q;

    rhs = f_loc[invalpha[alpha]];
    q = T(.5);

    f1 = f[i + j*nx + alpha*dec];
    f2 = f[i + j*nx + invalpha[alpha]*dec];

    //bounce back conditions
    if (type == 1)
    {
        std::cout<<"Condition 1"<<std::endl;
        if (q<=T(.5))
            f_loc[invalpha[alpha]] = (T(1.) - T(2.)*q)*f_loc[alpha] + T(2.)*q*f1 + rhs;
        else
            f_loc[invalpha[alpha]] = (T(1.) - T(.5)/q)*f2 +T(.5)/q*f1 + rhs;
    }
    //anti bounce back conditions
    else if (type == 2)
    {
        std::cout<<"Condition 2"<<std::endl;
        if (q<T(.5))
            f_loc[invalpha[alpha]] = -(T(1.) - T(2.)*q)*f_loc[alpha] - T(2.)*q*f1 + rhs;
        else
            f_loc[invalpha[alpha]] = -(T(1.) - T(.5)/q)*f2 -T(.5)/q*f1 + rhs;
    }
    //Neumann conditions
    else if (type == 3)
    {
        std::cout<<"Condition 3"<<std::endl;
        f_loc[invalpha[alpha]] = f2;
    }
}


template< typename T>
inline void apply_bc_scalar( std::vector<T> const & f
             , std::array<T,9> & f_loc
             , int bc
             , std::vector<int> const & alphaTab
             , int nx
             , int ny
             , int i
             , int j
             )
{
    int alpha = alphaTab[i+j*nx];

    for(int k=0;k<8;k++){
        if (alpha>>k&1){
            bouzidi_scalar(f, f_loc, T(0.), k+1, bc, nx, ny, i, j);
        }
    }
}

template< typename T>
inline void onetime_step_scalar(  std::vector<T> & f
                   ,std::vector<T> & fcopy
                   ,std::vector<int> & bc
                   ,std::vector<int> & alpha
                   ,std::array<T,6> const & s
                   ,int nx
                   ,int ny
                   ,int i
                   ,int j
                   ,std::vector<T> & m
                  )
{
    std::array<T,9> m_loc = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
    std::array<T,9> f_loc = {0,0,0,0,0,0,0,0,0};

    int bc_ = bc[ i + j*nx ];

    get_f_scalar(f, f_loc, nx, ny, i, j);
    apply_bc_scalar(f, f_loc, bc_, alpha, nx, ny, i, j);
    f2m_scalar(f_loc, m_loc);
    relaxation_scalar(m_loc,s);
    m2f_scalar(m_loc, f_loc);

    int dec = nx*ny;
    int ind = i +j*nx;

    for(int k=0;k<9;k++){
    m[ind] = m_loc[k];
    ind += dec;
    }

    set_f_scalar(fcopy, f_loc, nx, ny, i, j);
}

#endif
