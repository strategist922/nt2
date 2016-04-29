//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <nt2/include/functions/extent.hpp>
#include <nt2/sdk/memory/c_allocator.hpp>

#include <nt2/sdk/unit/tests.hpp>
#include <nt2/sdk/unit/module.hpp>

NT2_TEST_CASE_TPL( table_release ,NT2_TYPES)
{
  using nt2::table;

  typename nt2::table<T>::pointer ptr;

  {
    nt2::table<T> a(nt2::of_size(3, 3));

    for(int i=1; i <= 3; i++)
      for(int j=1; j <= 3; j++)
        a(i, j) = T(i + 10*j);

    ptr = a.release();

    NT2_TEST_EQUAL(nt2::extent(a), nt2::of_size(0));
  }

  for(int j=0; j < 3; j++)
    for(int i=0; i < 3; i++)
        NT2_TEST_EQUAL( ptr[i+3*j], T(i+1 + 10*(j+1)) );

  boost::simd::deallocate(ptr);
}

NT2_TEST_CASE_TPL( table_release_c_style ,NT2_TYPES)
{
  using nt2::table;
  using nt2::c_allocator;

  typename nt2::table<T, c_allocator<T> >::pointer ptr;

  {
    nt2::table<T,c_allocator<T> > a(nt2::of_size(3, 3));

    for(int i=1; i <= 3; i++)
      for(int j=1; j <= 3; j++)
        a(i, j) = T(i + 10*j);

    ptr = a.release();

    NT2_TEST_EQUAL(nt2::extent(a), nt2::of_size(0));
  }

  for(int j=0; j < 3; j++)
    for(int i=0; i < 3; i++)
        NT2_TEST_EQUAL( ptr[i+3*j], T(i+1 + 10*(j+1)) );

  ::free(ptr);
}
