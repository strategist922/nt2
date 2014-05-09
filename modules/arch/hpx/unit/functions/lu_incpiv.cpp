//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#define NT2_UNIT_MODULE "nt2 linalg toolbox - incpiv lu function"

#include <nt2/table.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/transpose.hpp>
#include <nt2/include/functions/display.hpp>

#include <nt2/include/functions/pgetrf_incpiv.hpp>
#include <nt2/include/functions/larnv.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>

#include <cstdio>

NT2_TEST_CASE_TPL ( getrf_incpiv, (double))
{
  std::size_t size = 8;
  std::size_t nb = 4;
  std::size_t ib = (nb<40) ? nb : 40;
  std::size_t ntiles = size/nb;

  nt2::table<T> A( nt2::of_size(size,size) );
  nt2::table<T> L( nt2::of_size(ib*ntiles,size) );
  nt2::table<nt2_la_int> ipiv( nt2::of_size(size,ntiles) );

  nt2::table<T> A1 =
  nt2::trans( nt2::cons<T>(nt2::of_size(size,size),
            0.120625,       0.767477,       0.310286,       0.352663,       0.738238,        0.700821,        0.698516,        0.683620,
            0.643846,       0.846837,       0.492237,       0.108638,       0.883349,        0.946291,        0.476169,        0.046268,
            0.062342,       0.168109,       0.037767,       0.873379,       0.309340,        0.465197,        0.150777,        0.070173,
            0.490279,       0.404544,       0.698914,       0.962905,       0.446327,        0.389050,        0.505525,        0.499435,
            0.306079,       0.302473,       0.170383,       0.533232,       0.040278,        0.438842,        0.813342,        0.299594,
            0.816414,       0.773004,       0.416681,       0.405641,       0.927313,        0.301427,        0.187764,        0.692897,
            0.997180,       0.315647,       0.119911,       0.850340,       0.753849,        0.844803,        0.380459,        0.051028,
            0.424599,       0.835469,       0.227429,       0.160437,       0.586095,        0.080179,        0.589011,        0.676343 ));


  nt2::table<T> A2 =
  nt2::trans( nt2::cons<T>(nt2::of_size(size,size),
            0.997180,       0.315647,       0.119911,       0.850340,        0.753849,       0.844803,        0.380459,        0.051028,
            0.187350,       0.701067,       0.176371,      -0.201637,       -0.000261,      -0.124892,        0.383430,        0.730615,
            0.096827,       0.141440,       0.410156,       1.011350,        0.265106,      -0.279538,        0.427011,        0.654616,
            0.761485,      -0.394713,      -0.099324,       0.916309,        0.169131,       0.271757,        0.044325,       -0.868140,
            0.374906,       0.020810,       0.023475,       0.382510,       -0.390214,       0.217553,        0.060904,        0.035248,
            0.788627,       0.389646,       0.191786,      -0.583545,        0.059635,       0.839321,        0.184727,        0.010882,
            0.818722,       0.733990,       0.460928,      -0.664305,       -0.270862,       0.218330,       -0.617346,       -0.100506,
            0       ,       0.868422,       0.158234,       0.379115,        0       ,       0.838542,       -0.872145,       -0.101955 ));


  nt2::table<nt2_la_int> ISEED = nt2::cons<nt2_la_int>(0,0,0,1);

  nt2::larnv(1,
             boost::proto::value( ISEED ),
             boost::proto::value( A )
            );

  NT2_TEST_ULP_EQUAL( A, A1, 1e11);

  nt2::pgetrf_incpiv(nb,A,L,ipiv);

  // nt2::display(A);
  // NT2_TEST_ULP_EQUAL( A, A2, double(1e-5));
}
