//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <vector>
#include <array>
#include <iostream>

#include <nt2/linalg/details/blas/mm.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/unit/details/prng.hpp>
#include <nt2/sdk/bench/metric/absolute_time.hpp>
#include <nt2/sdk/bench/metric/speedup.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/protocol/until.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include "D2Q9_kernels.hpp"

using namespace nt2;
using namespace nt2::bench;


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


  void operator()()
  {
    int max_steps = 10;
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
            onetime_step<T>
            (*fin, *fout, bc, alpha, s, nx, ny, i_, j_);
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

  void relaxation(std::array<T,6> const s_, T const rho, T const la)
  {
    T dummy_ = T(1.)/(la*la*rho);

    for(int i = 0; i < nx; i++)
    {
     for(int j = 0; j < ny; j++)
     {
      T qx2 = dummy_*m_(1,i,j)*m_(1,i,j);
      T qy2 = dummy_*m_(2,i,j)*m_(2,i,j);
      T q2  = qx2 + qy2;
      T qxy = dummy_*m_(1,i,j)*m_(2,i,j);

      m_(3,i,j) = m_(3,i,j)*(1-s_[0]) + s_[0]*(-2.*m_(0,i,j) + T(3.)*q2);
      m_(4,i,j) = m_(4,i,j)*(1-s_[1]) + s_[1]*(m_(0,i,j) + T(1.5)*q2);
      m_(5,i,j) = m_(5,i,j)*(1-s_[2]) - s_[2]*m_(1,i,j)/la;
      m_(6,i,j) = m_(6,i,j)*(1-s_[3]) - s_[3]*m_(2,i,j)/la;
      m_(7,i,j) = m_(7,i,j)*(1-s_[4]) + s_[4]*(qx2-qy2);
      m_(8,i,j) = m_(8,i,j)*(1-s_[5]) + s_[5]*qxy;
    }
   }
  }

void m2f(T const la)
{
  T a(1./9)
  , b(1./36.)
  , c(1./(6.*la))
  , d(1./12)
  , e(1./4.);

  int nine   = 9;
  int bound  = nx*ny;

  T one  = 1.;
  T zero = 0.;

  std::vector<T> invM
  =  {  a,  0,  0, -4*b,  4*b,    0,    0,  0,  0,
    a,  c,  0,   -b, -2*b, -2*d,    0,  e,  0,
    a,  0,  c,   -b, -2*b,    0, -2*d, -e,  0,
    a, -c,  0,   -b, -2*b,  2*d,    0,  e,  0,
    a,  0, -c,   -b, -2*b,    0,  2*d, -e,  0,
    a,  c,  c,  2*b,    b,    d,    d,  0,  e,
    a, -c,  c,  2*b,    b,   -d,    d,  0, -e,
    a, -c, -c,  2*b,    b,   -d,   -d,  0,  e,
    a,  c, -c,  2*b,    b,    d,   -d,  0, -e
  };

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
, s({s3,s4,s5,s6,s7,s8})
, bc(nx*ny,0)
, m (9 * nx * ny, 0.)
, f (9 * nx * ny, 0.)
, fcopy(9 * nx * ny)
, rhs  (9 * nx * ny, 0.)
, alpha (nx * ny, 0)
, s1x((posx_obs - l_obs/2)/dx), s2x((posx_obs + l_obs/2)/dx)
, s1y((posy_obs - L_obs/2)/dx), s2y((posy_obs + L_obs/2)/dx)
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

  relaxation({1,1,1,1,1,1},rhoo, 1.);

  m2f(1.);

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

  std::array<T,6> s;

// Set boundary conditions outside the domain and on the rectangular obstacle
  std::vector<int> bc;

  std::vector<T> m
  ,f,fcopy
  ,rhs;

  std::vector<int> alpha;

  int s1x, s2x
  ,s1y, s2y;
};

NT2_REGISTER_BENCHMARK_TPL( latticeboltzmann_scalar, (float) )
{
  run_until_with< latticeboltzmann_scalar<T> > ( 3., 1
                                  , fixed(1024)
                                  , absolute_time<stats::median_>()
                                  );
}
