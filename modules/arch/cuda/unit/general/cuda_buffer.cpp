//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/memory/cuda/buffer.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>

NT2_TEST_CASE_TPL(cuda_buffer_default, NT2_REAL_TYPES)
{
  nt2::memory::cuda_buffer<T> b;

  NT2_TEST(b.empty());
  NT2_TEST_EQUAL(b.size()     , 0u      );
  NT2_TEST_EQUAL(b.begin()    , b.end() );
}

NT2_TEST_CASE_TPL(cuda_buffer_size_init, NT2_REAL_TYPES)
{
  nt2::memory::cuda_buffer<T> b(5);

  NT2_TEST(!b.empty());
  NT2_TEST_EQUAL(b.size()     , 5u    );
}


NT2_TEST_CASE_TPL(cuda_buffer_resize, NT2_REAL_TYPES)
{
  nt2::memory::cuda_buffer<T> b(5);
  b.resize(15);

  NT2_TEST(!b.empty());
  NT2_TEST_EQUAL(b.size()     , 15u    );
}

