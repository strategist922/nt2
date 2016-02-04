//==============================================================================
//            Copyright 2016   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <nt2/include/functions/slice_of.hpp>
#include <nt2/include/functions/ones.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>

NT2_TEST_CASE( slice_1D )
{
  for(int i=1;i<=3*5;++i)
  {
    nt2::table<double> x( nt2::of_size(16,3,5) );

    auto s = nt2::slice_of(x,i);
    NT2_TEST_EQUAL( nt2::extent(s), nt2::of_size(16) );

    s = nt2::ones(16,1);

    NT2_TEST_EQUAL( x(nt2::_,i), nt2::ones(16,1) );
  }
}

NT2_TEST_CASE( slice_2D )
{
  for(int j=1;j<=5;++j)
    for(int i=1;i<=3;++i)
    {
      nt2::table<double> x( nt2::of_size(16,3,5) );

      auto s = nt2::slice_of(x,i,j);
      NT2_TEST_EQUAL( nt2::extent(s), nt2::of_size(16) );

      s = nt2::ones(16,1);

      NT2_TEST_EQUAL( x(nt2::_,i,j), nt2::ones(16,1) );
    }
}

NT2_TEST_CASE( slice_3D )
{
  for(int k=1;k<=2;++k)
    for(int j=1;j<=5;++j)
      for(int i=1;i<=3;++i)
      {
        nt2::table<double> x( nt2::of_size(16,3,5,2) );

        auto s = nt2::slice_of(x,i,j,k);
        NT2_TEST_EQUAL( nt2::extent(s), nt2::of_size(16) );

        s = nt2::ones(16,1);

        NT2_TEST_EQUAL( x(nt2::_,i,j,k), nt2::ones(16,1) );
      }
}
