//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/table.hpp>

#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/mtimes.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/reshape.hpp>

#include <nt2/sdk/meta/as.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>

//#include <nt2/sdk/bench/benchmark.hpp>
//#include <nt2/sdk/unit/details/prng.hpp>
//#include <nt2/sdk/bench/metric/absolute_time.hpp>
//#include <nt2/sdk/bench/metric/speedup.hpp>
//#include <nt2/sdk/bench/setup/fixed.hpp>
//#include <nt2/sdk/bench/protocol/until.hpp>
//#include <nt2/sdk/bench/stats/median.hpp>

#include <iostream>

#include "D2Q9_kernels.hpp"

using namespace nt2;
//using namespace nt2::bench;

template<typename T> struct latticeboltzmann_nt2_opt
{
  inline void onetime_step( nt2::table<T> & in
                          , nt2::table<T> & out
                          )
  {
    get_f(in,out,nx,ny);
    apply_bc(in, out, bc, alpha, nx, ny);
    f2m_m2f(out, m, nx, ny, invF);
    relaxation(m,s);
    f2m_m2f(m, out, nx, ny, invM);
  }

  void operator()()
  {
    int max_steps = 1;

    nt2::table<T> * fout = &fcopy;
    nt2::table<T> * fin  = &f;


    for(int step = 0; step<max_steps; step++)
    {
        onetime_step(*fin, *fout);

        std::swap(fout,fin);
     }

   }

  friend std::ostream& operator<<(std::ostream& os, latticeboltzmann_nt2_opt<T> const& p)
  {
    return os << "(" << p.size()<< ")";
  }

  int size() const { return nx*ny; }

  latticeboltzmann_nt2_opt(int size_)
  :  nx(size_),ny(size_/2)
  , Longueur(2.), Largeur(1.)
  , xmin(0.0), xmax(Longueur), ymin(-0.5*Largeur), ymax(0.5*Largeur)
  , dx(Longueur/nx)
  , vit_schema(1)
  , dt(dx/vit_schema)
  , L_obs(0.25), l_obs(0.05)
  , posx_obs(0.25), posy_obs(0.5)
  , Re(100.0)
  , max_velocity(0.05)
  , rhoo(1.)
  , mu(rhoo*max_velocity*L_obs/Re)
  , zeta(3.*mu)
  , dummy(3.0/(vit_schema*vit_schema*rhoo*dt))
  , s3(1.0/(0.5+zeta*dummy))
  , s4(s3)
  , s5(s4)
  , s6(s4)
  , s7(1.0/(0.5+mu*dummy))
  , s8(s7)
  , s     (nt2::cons(s3,s4,s5,s6,s7,s8))
  , bc    (nt2::of_size(nx, ny))
  , m     (nt2::of_size(nx, ny, 9))
  , f     (nt2::of_size(nx, ny, 9))
  , fcopy (nt2::of_size(nx, ny, 9))
  , rhs   (nt2::of_size(nx, ny, 9))
  , alpha (nt2::of_size(nx, ny))
  , s1x(1 + (posx_obs - l_obs/2)/dx), s2x(1 + (posx_obs + l_obs/2)/dx)
  , s1y(1 + (posy_obs - L_obs/2)/dx), s2y(1 + (posy_obs + L_obs/2)/dx)
  , la (1.)
  , a (1./9.)
  , b (1./36.)
  , c (1./(6.*la))
  , d (1./12.)
  , e (.25)
  , invF (   nt2::cons<T>( nt2::of_size(9 ,9),
                1,  1,  1,  1,  1,  1,  1,  1,  1,
                0, la,  0,-la,  0, la,-la,-la, la,
                0,  0, la,  0,-la, la, la,-la,-la,
               -4, -1, -1, -1, -1,  2,  2,  2,  2,
                4, -2, -2, -2, -2,  1,  1,  1,  1,
                0, -2,  0,  2,  0,  1, -1, -1,  1,
                0,  0, -2,  0,  2,  1,  1, -1, -1,
                0,  1, -1,  1, -1,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  1, -1,  1, -1
          ))
  , invM (nt2::cons<T>( nt2::of_size(9 ,9),
          a,  0,  0, -4*b,  4*b,    0,    0,  0,  0,
          a,  c,  0,   -b, -2*b, -2*d,    0,  e,  0,
          a,  0,  c,   -b, -2*b,    0, -2*d, -e,  0,
          a, -c,  0,   -b, -2*b,  2*d,    0,  e,  0,
          a,  0, -c,   -b, -2*b,    0,  2*d, -e,  0,
          a,  c,  c,  2*b,    b,    d,    d,  0,  e,
          a, -c,  c,  2*b,    b,   -d,    d,  0, -e,
          a, -c, -c,  2*b,    b,   -d,   -d,  0,  e,
          a,  c, -c,  2*b,    b,    d,   -d,  0, -e
          ))
  {
    nt2::table<T> s_init = nt2::ones(6,nt2::meta::as_<T>());

    bc    = nt2::zeros(nt2::of_size(nx, ny), nt2::meta::as_<int>());
    alpha = nt2::zeros(nt2::of_size(nx, ny), nt2::meta::as_<int>());;
    m     = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());
    f     = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());
    fcopy = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());
    rhs   = nt2::zeros(nt2::of_size(nx, ny, 9), nt2::meta::as_<T>());

    // Set boundary conditions outside the domain and on the rectangular obstacle
    bc(1,_)   = 1;
    bc(nx,_)  = 3;
    bc(_, 1)  = 1;
    bc(_, ny) = 1;

    m(_,_,1) = rhoo;
    m(_,_,2) = rhoo*max_velocity;

    relaxation( m
              , s_init
              );

    f2m_m2f(m,f,nx,ny,invM);

    bc(_(s1x,s2x-1),_(s1y,s2y-1)) = 1;

    // Set in binary format which velocity is involved on the boundary
    // Example 01001: we have the velocity 1 and 4 on the boundary

    // on the rectangular obstacle
    // moment 1
    alpha(s1x-1, _(s1y,s2y-1)) += 1<<0;
    // moment 2
    alpha(_(s1x,s2x-1), s1y-1) += 1<<1;
    // moment 3
    alpha(s2x, _(s1y,s2y-1))   += 1<<2;
    // moment 4
    alpha(_(s1x,s2x-1),s2y)    += 1<<3;
    // // moment 5
    alpha(_(s1x-1,s2x-2),s1y-1) += 1<<4;
    alpha(s1x-1, _(s1y,s2y-2))  += 1<<4;
    // moment 6
    alpha(_(s1x+1,s2x),s1y-1) += 1<<5;
    alpha(s2x,_(s1y,s2y-2))   += 1<<5;
    // moment 7
    alpha(s2x,_(s1y+1,s2y))   += 1<<6;
    alpha(_(s1x+1,s2x-1),s2y) += 1<<6;
    // moment 8
    alpha(s1x-1,_(s1y+1,s2y)) += 1<<7;
    alpha(_(s1x,s2x-2),s2y)   += 1<<7;

    // outside the domain
    alpha(2,_) = 1<<5 | 1<<2 | 1<<6;
    alpha(nx-1,_) += 1<<4 | 1<<0 | 1<<7;
    alpha(_,2) += 1<<3;
    alpha(_,ny-1) += 1<<1;
    alpha(_(3,nx-2),2) += 1<<6 | 1<<7;
    alpha(_(3,nx-2),ny-1) += 1<<5 | 1<<4;

    // Set rhs on the boundary
    // right border
    f(nx,_,4)          += -f(nx-1,_,2);
    f(nx,_(1,ny-1),7)  += -f(nx-1,_(2,ny),9);
    f(nx,_(2,ny),8)    += -f(nx-1,_(1,ny-1),6);
    // left border
    f(1,_,2)           += -f(2,_,4);
    f(1,_(2,ny),9)     += -f(2,_(1,ny-1),7);
    f(1,_(1,ny-1),6)   += -f(2,_(2,ny),8);
    // // top border
    f(_,ny,5)          += -f(_,ny-1,3);
    f(_(2,nx-1),ny,8)  += -f(_(1,nx-2),ny-1,6);
    f(_(2,nx-1),ny,9)  += -f(_(3,nx),ny-1,7);
    // bottom border
    f(_,1,3)           += -f(_,2,5);
    f(_(2,nx-1),1,6)   += -f(_(3,nx),2,8);
    f(_(2,nx-1),1,7)   += -f(_(1,nx-2),2,9);
    // rectangular obstacle
    f(_(s1x,s2x-1),_(s1y,s2y-1),_) = T(0);
  }

  // Domain, space and time step
  int nx, ny;

  T Longueur, Largeur
   ,xmin, xmax, ymin, ymax
   ,dx = Longueur/nx
   ,vit_schema
   ,dt;

  // Obstacle
  T L_obs, l_obs
   ,posx_obs, posy_obs;

// Physical parameters
  T Re
   ,max_velocity
   ,rhoo
   ,mu
   ,zeta
   ,dummy
   ,s3
   ,s4
   ,s5
   ,s6
   ,s7
   ,s8;

   nt2::table<T> s;

// Set boundary conditions outside the domain and on the rectangular obstacle
   nt2::table<int> bc;

   nt2::table<T>  m
                 ,f,fcopy
                 ,rhs;

   nt2::table<int> alpha;

   int s1x, s2x
      ,s1y, s2y;

   T la, a, b, c, d, e;

   nt2::table<T> invF, invM;
};

/*****************************************************************************/

template<typename T> struct latticeboltzmann_scalar
{
  void display_f()
  {
    std::cout.precision(3);
    for(int k = 0; k<9; k++){
    std::cout<<"f(:,:,"<<k<<") =\n[";
    for(int i = 0; i<nx; i++){

      std::cout<<"f("<<i<<",:,"<<k<<") =\n[";
      for(int j = 0; j<ny; j++){

        std::cout<< f_(k, i, j)<<", ";
      }
      std::cout<<"]\n";
    }
    std::cout<<"]\n";
  }
  }

  void display_fcopy()
  {
    std::cout.precision(3);
    for(int k = 0; k<9; k++){
    std::cout<<"fcopy(:,:,"<<k<<") =\n[";
    for(int i = 0; i<nx; i++){

      std::cout<<"fcopy("<<i<<",:,"<<k<<") =\n[";
      for(int j = 0; j<ny; j++){

        std::cout<< fcopy_(k, i, j)<<", ";
      }
      std::cout<<"]\n";
    }
    std::cout<<"]\n";
  }
  }

  void display_alpha()
  {
    std::cout.precision(3);
    for(int i = 0; i<nx; i++){
      std::cout<<"alpha("<<i<<",:) =\n[";
      for(int j = 0; j<ny; j++){

        std::cout<< alpha_(i, j)<<", ";
      }
      std::cout<<"]\n";
    }
  }

  inline void onetime_step(  std::vector<T> & in
                           , std::vector<T> & out
                           , int i
                           , int j
                          )
  {
      int condition = bc[ i + j*nx ];

      get_f_scalar(in, f_loc, nx, ny, i, j);
      apply_bc_scalar(in, f_loc, condition, alpha, nx, ny, i, j);
      f2m_m2f_scalar(f_loc, m_loc, invF);
      relaxation_scalar(m_loc,s);
      f2m_m2f_scalar(m_loc, f_loc, invM);
      set_f_scalar(out, f_loc, nx, ny, i, j);
  }

  void operator()()
  {
    int max_steps = 1;
    int bi = 128;
    int bj = 1;

    std::vector<T> * fout = &fcopy;
    std::vector<T> * fin  = &f;


    for(int step = 0; step<max_steps; step++)
    {
     for(int j = 0; j<ny; j+=bj)
     {
       for(int i = 0; i<nx; i+=bi)
       {
        int chunk_i =  (i <= nx-bi) ? bi : nx-i;
        int chunk_j =  (j <= ny-bj) ? bj : ny-j;
        int max_i = i+chunk_i;
        int max_j = j+chunk_j;

        for(int j_ = j; j_<max_j; j_++)
          for(int i_ = i; i_<max_i; i_++)
          {
            onetime_step(*fin, *fout,i_, j_);
          }
        }
      }

      std::swap(fout,fin);
    }
  }

  friend std::ostream& operator<<(std::ostream& os, latticeboltzmann_scalar<T> const& p)
  {
    return os << "(" << p.size()<< ")";
  }

  int size() const { return nx*ny; }

  void pre_relaxation(std::vector<T> s_, T const rho)
  {
    T dummy_ = T(1.)/(la*la*rho);

    for(int i = 0; i < nx; i++)
    {
     for(int j = 0; j < ny; j++)
     {
      m_(3,i,j) = m_(3,i,j)*(1-s_[0])
                + s_[0]*(-2.*m_(0,i,j)
                        + T(3.)*(dummy_*m_(1,i,j)*m_(1,i,j)
                                +dummy_*m_(2,i,j)*m_(2,i,j)
                                )
                        );

      m_(4,i,j) = m_(4,i,j)*(1-s_[1])
                + s_[1]*( m_(0,i,j)
                        + T(1.5)*(dummy_*m_(1,i,j)*m_(1,i,j)
                                 +dummy_*m_(2,i,j)*m_(2,i,j)
                                 )
                        );

      m_(5,i,j) = m_(5,i,j)*(1-s_[2]) - s_[2]*m_(1,i,j)/la;
      m_(6,i,j) = m_(6,i,j)*(1-s_[3]) - s_[3]*m_(2,i,j)/la;

      m_(7,i,j) = m_(7,i,j)*(1-s_[4])
                + s_[4]*(dummy_*m_(1,i,j)*m_(1,i,j)
                        -dummy_*m_(2,i,j)*m_(2,i,j)
                        );

      m_(8,i,j) = m_(8,i,j)*(1-s_[5]) + s_[5]*dummy_*m_(1,i,j)*m_(2,i,j);
    }
   }
  }

void pre_m2f()
{
  int nine   = 9;
  int bound  = nx*ny;

  T one  = 1.;
  T zero = 0.;

// Row Major Matrix-Matrix multiplication with Column Major Blas
  nt2::details::
  gemm( "N", "N"
    , &bound, &nine, &nine
    , &one
    , & m[0], &bound
    , & invM[0], &nine
    , &zero
    , &f[0], &bound
    );
}

inline int& alpha_(int const i, int const j)
{
  return alpha[i + nx*j];
}

inline int const & alpha_(int const i, int const j) const
{
  return alpha[i + nx*j];
}

inline T& f_(int const k, int const i, int const j)
{
  return f[ nx*ny*k + i + nx*j];
}

inline T const & f_(int const k, int const i, int const j) const
{
  return f[ nx*ny*k + i + nx*j];
}

inline T& fcopy_(int const k, int const i, int const j)
{
  return fcopy[ nx*ny*k + i + nx*j];
}

inline T const & fcopy_(int const k, int const i, int const j) const
{
  return fcopy[ nx*ny*k + i + nx*j];
}

inline int & bc_(int const i, int const j)
{
  return bc[i + nx*j];
}

inline int const & bc_(int const i, int const j) const
{
  return bc[i + nx*j];
}

inline T& m_(int const k, int const i, int const j)
{
  return m[ nx*ny*k + i + nx*j];
}

inline T const & m_(int const k, int const i, int const j) const
{
  return m[ nx*ny*k + i + nx*j];
}

latticeboltzmann_scalar(int size_)
: nx(size_),ny(size_/2)
, Longueur(2.), Largeur(1.)
, xmin(0.0), xmax(Longueur), ymin(-0.5*Largeur), ymax(0.5*Largeur)
, dx(Longueur/nx)
, vit_schema(1)
, dt(dx/vit_schema)
, L_obs(0.25), l_obs(0.05)
, posx_obs(0.25), posy_obs(0.5)
, Re(100.0)
, max_velocity(0.05)
, rhoo(1.)
, mu(rhoo*max_velocity*L_obs/Re)
, zeta(3.*mu)
, dummy(3.0/(vit_schema*vit_schema*rhoo*dt))
, s3(1.0/(0.5+zeta*dummy))
, s4(s3)
, s5(s4)
, s6(s4)
, s7(1.0/(0.5+mu*dummy))
, s8(s7)
, s({s3,s4,s5,s6,s7,s8})
, bc(nx*ny,0)
, m (9 * nx * ny, 0.)
, f (9 * nx * ny, 0.)
, fcopy(9 * nx * ny)
, rhs  (9 * nx * ny, 0.)
, alpha (nx * ny, 0)
, s1x((posx_obs - l_obs/2)/dx), s2x((posx_obs + l_obs/2)/dx)
, s1y((posy_obs - L_obs/2)/dx), s2y((posy_obs + L_obs/2)/dx)
, m_loc(9,0.)
, f_loc(9,0.)
, la (1.)
, a (1./9.)
, b (1./36.)
, c (1./(6.*la))
, d (1./12.)
, e (.25)
, invF(
       {
        1,  1,  1,  1,  1,  1,  1,  1,  1,
        0, la,  0,-la,  0, la,-la,-la, la,
        0,  0, la,  0,-la, la, la,-la,-la,
       -4, -1, -1, -1, -1,  2,  2,  2,  2,
        4, -2, -2, -2, -2,  1,  1,  1,  1,
        0, -2,  0,  2,  0,  1, -1, -1,  1,
        0,  0, -2,  0,  2,  1,  1, -1, -1,
        0,  1, -1,  1, -1,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  1, -1,  1, -1
       }
     )
 , invM(
       {
        a,  0,  0, -4*b,  4*b,    0,    0,  0,  0,
        a,  c,  0,   -b, -2*b, -2*d,    0,  e,  0,
        a,  0,  c,   -b, -2*b,    0, -2*d, -e,  0,
        a, -c,  0,   -b, -2*b,  2*d,    0,  e,  0,
        a,  0, -c,   -b, -2*b,    0,  2*d, -e,  0,
        a,  c,  c,  2*b,    b,    d,    d,  0,  e,
        a, -c,  c,  2*b,    b,   -d,    d,  0, -e,
        a, -c, -c,  2*b,    b,   -d,   -d,  0,  e,
        a,  c, -c,  2*b,    b,    d,   -d,  0, -e
       }
  )
{

    // Set boundary conditions outside the domain and on the rectangular obstacle
  for(int j = 0; j<ny; j++)
    bc_(0, j)  = 1;
  for(int j = 0; j<ny; j++)
    bc_(nx-1, j) = 3;
  for(int i = 0; i<nx; i++)
    bc_(i, 0)  = 1;
  for(int i = 0; i<nx; i++)
    bc_(i, ny-1) = 1;

  for(int i = 0; i<nx; i++)
   for(int j = 0; j<ny; j++)
   {
    m_(0, i, j) = rhoo;
    m_(1, i, j) = rhoo*max_velocity;
  }

  pre_relaxation({1,1,1,1,1,1},rhoo);

  pre_m2f();

  for(int i = s1x; i<s2x; i++)
   for(int j = s1y; j<s2y; j++)
   {
    bc_(i, j) = 1;
   }

    // on the rectangular obstacle
    // moment 1
  for(int j = s1y; j<s2y; j++)
    alpha_(s1x-1, j) += 1<<0;

    // moment 2
  for(int i = s1x; i<s2x; i++)
    alpha_(i, s1y-1) += 1<<1;

    // moment 3
  for(int j = s1y; j<s2y; j++)
   alpha_(s2x, j) += 1<<2;

    // moment 4
  for(int i = s1x; i<s2x; i++)
   alpha_(i, s2y) += 1<<3;

    // moment 5
  for(int i = s1x-1; i<s2x-1; i++)
    alpha_(i, s1y-1) += 1<<4;
  for(int j = s1y; j<s2y-1; j++)
    alpha_(s1x-1, j) += 1<<4;

    // moment 6
  for(int i = s1x+1; i<s2x+1; i++)
    alpha_(i, s1y-1) += 1<<5;
  for(int j = s1y; j<s2y-1; j++)
    alpha_(s2x, j) += 1<<5;

    // moment 7
  for(int j = s1y+1; j<s2y+1; j++)
    alpha_(s2x, j) += 1<<6;
  for(int i = s1x+1; i<s2x; i++)
    alpha_(i, s2y) += 1<<6;

    // moment 8
  for(int j = s1y+1; j<s2y+1; j++)
    alpha_(s1x-1, j) += 1<<7;
  for(int i = s1x; i<s2x-1; i++)
    alpha_(i, s2y) += 1<<7;

    // outside the domain
  for(int j = 0; j<ny; j++)
    alpha_(1, j) = 1<<5 | 1<<2 | 1<<6;

  for(int j = 0; j<ny; j++)
    alpha_(nx-2, j) += 1<<4 | 1<<0 | 1<<7;

  for(int i = 0; i<nx; i++)
    alpha_(i, 1) += 1<<3;

  for(int i = 0; i<nx; i++)
    alpha_(i, ny -2) += 1<<1;

  for(int i = 2; i<nx-2; i++)
    alpha_(i, 1) += 1<<6 | 1<<7;

  for(int i = 2; i<nx-2; i++)
    alpha_(i,ny -2) += 1<<5 | 1<<4;

    // Set rhs on the boundary

    // right border
  for(int j = 0; j<ny; j++)
    f_(3, nx-1, j) += -f_(1, nx-2, j);

  for(int j = 0; j<ny-1; j++)
    f_(6, nx-1, j) += -f_(8, nx-2, j+1);

  for(int j = 1; j<ny; j++)
    f_(7, nx-1, j) += -f_(5, nx -2, j-1);

      // left border
  for(int j = 0; j<ny; j++)
    f_(1, 0, j) += -f_(3, 1, j);

  for(int j = 1; j<ny; j++)
    f_(8, 0, j) += -f_(6, 1,j-1);

  for(int j = 0; j<ny-1; j++)
    f_(5, 0, j) += -f_(7, 1, j+1);

      // top border
  for(int i = 0; i<nx; i++)
    f_(4, i, ny-1) += -f_(2, i, ny-2);

  for(int i = 1; i<nx-1; i++)
    f_(7, i, ny-1) += -f_(5, i-1, ny-2);

  for(int i = 1; i<nx-1; i++)
    f_(8, i, ny-1) += -f_(6, i+1, ny-2);

      // bottom border
  for(int i = 0; i<nx; i++)
    f_(2, i, 0) += -f_(4, i, 1);

  for(int i = 1; i<nx-1; i++)
    f_(5, i, 0) += -f_(7, i+1, 1);

  for(int i = 1; i<nx-1; i++)
    f_(6, i, 0) += -f_(8, i-1, 1);

      // rectangular obstacle
  for(int k = 0; k<9; k++)
    for(int i = s1x; i<s2x; i++)
      for(int j = s1y; j<s2y; j++)
        f_(k, i, j) = 0;
  }

private:

  // Domain, space and time step
  int nx, ny;

  T Longueur, Largeur
  ,xmin, xmax, ymin, ymax
  ,dx = Longueur/nx
  ,vit_schema
  ,dt;

  // Obstacle
  T L_obs, l_obs
  ,posx_obs, posy_obs;

// Physical parameters
  T Re
  ,max_velocity
  ,rhoo
  ,mu
  ,zeta
  ,dummy
  ,s3
  ,s4
  ,s5
  ,s6
  ,s7
  ,s8;

  std::vector<T> s;

// Set boundary conditions outside the domain and on the rectangular obstacle
  std::vector<int> bc;

  std::vector<T> m
  ,f,fcopy
  ,rhs;

  std::vector<int> alpha;

  int s1x, s2x
  ,s1y, s2y;

  std::vector<T> m_loc;
  std::vector<T> f_loc;

  T la, a, b, c, d, e;

  std::vector<T> invF;
  std::vector<T> invM;
};

//NT2_REGISTER_BENCHMARK_TPL( latticeboltzmann_nt2_opt, (float) )
//{
//  run_until_with< latticeboltzmann_nt2_opt<T> > ( 10., 10
//                                  , fixed(1024)
//                                  , absolute_time<stats::median_>()
//                                  );
//}

NT2_TEST_CASE( latticeboltzmann_test )
{
  latticeboltzmann_nt2_opt<float> test_nt2(16);
  latticeboltzmann_scalar<float> test_scalar(16);
  test_nt2();
  test_scalar();

  // for(int i = 0; i<16; i++)
  // for(int j = 0; j<8; j++)
  // {
  //  NT2_TEST_EQUAL(test_nt2.alpha(i+1,j+1)
  //                ,test_scalar.alpha_(i,j)
  //                );
  // }

  // for(int i = 0; i<16; i++)
  // for(int j = 0; j<8; j++)
  // {
  //  NT2_TEST_EQUAL(test_nt2.bc(i+1,j+1)
  //                ,test_scalar.bc_(i,j)
  //                );
  // }

  // for(int k = 0; k<9; k++)
  // for(int i = 0; i<16; i++)
  // for(int j = 0; j<8; j++)
  // {
  //  NT2_TEST_EQUAL(test_nt2.m(i+1,j+1,k+1)
  //                ,test_scalar.m_(k,i,j)
  //                );
  // }

  for(int k = 0; k<9; k++)
  for(int i = 0; i<16; i++)
  for(int j = 0; j<8; j++)
  {
   NT2_TEST_EQUAL(test_nt2.fcopy(i+1,j+1,k+1)
                 ,test_scalar.fcopy_(k,i,j)
                 );
  }

}
