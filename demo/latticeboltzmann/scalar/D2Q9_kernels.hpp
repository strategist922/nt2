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
#include <vector>
#include <iostream>

template< typename T>
inline void relaxation(std::vector<T> & m, std::vector<T> const & s)
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
void get_f( std::vector<T> const & f
          , std::vector<T> & f_loc
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
inline void f2m_m2f(std::vector<T> const & in
                   ,std::vector<T> & out
                   ,std::vector<T> & inv
                   )
{
  T one  = T(1.);
  T zero = T(0.);

  int inc    = 1;
  int nine   = 9;

// Row Major Matrix-Matrix multiplication with Column Major Blas
  nt2::details::
  gemm( "T", "N"
    , &nine, &inc, &nine
    , &one
    , & inv[0], &nine
    , & in[0], &nine
    , &zero
    , &out[0], &nine
    );
}

template< typename T>
inline void set_f( std::vector<T> & f
          , std::vector<T> const & f_loc
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
inline void bouzidi( std::vector<T> const & f
            , std::vector<T> & f_loc
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
    std::vector<int> invalpha={0, 3, 4, 1, 2, 7, 8, 5, 6};
    T f1, f2, q;

    rhs = f_loc[invalpha[alpha]];
    q = T(.5);

    f1 = f[i + j*nx + alpha*dec];
    f2 = f[i + j*nx + invalpha[alpha]*dec];

    //bounce back conditions
    if (type == 1)
    {
        if (q<=T(.5))
            f_loc[invalpha[alpha]] = (T(1.) - T(2.)*q)*f_loc[alpha] + T(2.)*q*f1 + rhs;
        else
            f_loc[invalpha[alpha]] = (T(1.) - T(.5)/q)*f2 +T(.5)/q*f1 + rhs;
    }
    //anti bounce back conditions
    else if (type == 2)
    {
        if (q<T(.5))
            f_loc[invalpha[alpha]] = -(T(1.) - T(2.)*q)*f_loc[alpha] - T(2.)*q*f1 + rhs;
        else
            f_loc[invalpha[alpha]] = -(T(1.) - T(.5)/q)*f2 -T(.5)/q*f1 + rhs;
    }
    //Neumann conditions
    else if (type == 3)
    {
        f_loc[invalpha[alpha]] = f2;
    }
}


template< typename T>
inline void apply_bc( std::vector<T> const & f
             , std::vector<T> & f_loc
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
            bouzidi(f, f_loc, T(0.), k+1, bc, nx, ny, i, j);
        }
    }
}

#endif
