//==============================================================================
//         Copyright 2015          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/include/functions/ifft.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/table.hpp>

#include <nt2/sdk/unit/tests.hpp>
#include <nt2/sdk/unit/module.hpp>

#include <boost/dispatch/meta/as.hpp>
#include <complex>

NT2_TEST_CASE_TPL( ifft_real_complex, (double)(float))
{
  T Fs = 1000;
  T Period = 1./Fs;
  int L = 1024;

  nt2::table<T, nt2::_1D> in = nt2::_(T(1),T(10));

  nt2::table<std::complex<T>, nt2::_1D > out = nt2::ifft(in);

  nt2::table<std::complex<T> > ref
      = nt2::cons<std::complex<T> > ( nt2::of_size(1,10)
                                    , std::complex<T>( 5.5,  0.0               )
                                    , std::complex<T>(-0.5, -1.538841768587627 )
                                    , std::complex<T>(-0.5, -0.688190960235587 )
                                    , std::complex<T>(-0.5, -0.363271264002680 )
                                    , std::complex<T>(-0.5, -0.162459848116453 )
                                    , std::complex<T>(-0.5,  0.0               )
                                    , std::complex<T>(-0.5,  0.162459848116453 )
                                    , std::complex<T>(-0.5,  0.363271264002680 )
                                    , std::complex<T>(-0.5,  0.688190960235587 )
                                    , std::complex<T>(-0.5,  1.538841768587627 )
                                    );

  NT2_TEST_ULP_EQUAL(out, ref, 1 );
}
