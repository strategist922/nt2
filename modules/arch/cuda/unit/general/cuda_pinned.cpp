//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/memory/cuda/buffer.hpp>
#include <nt2/table.hpp>

#include <nt2/sdk/meta/type_id.hpp>
#include <nt2/sdk/meta/device.hpp>
#include <nt2/include/functions/two.hpp>
#include <nt2/include/functions/ones.hpp>
#include <cublas.h>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>

NT2_TEST_CASE_TPL(cuda_buffer_pinned, (double) )
{
  nt2::table<T,nt2::pinned_> A = nt2::ones(nt2::of_size(5,5), nt2::meta::as_<T>());
  nt2::table<T> B = A;

  NT2_TEST_EQUAL( (std::is_same<typename decltype(A)::allocator_type,nt2::memory::cuda_pinned_<T> >::value) , true  );
  NT2_TEST_EQUAL( nt2::meta::is_on_host<decltype(A)>::value , true );
  NT2_TEST_EQUAL(A , B );
}

NT2_TEST_CASE_TPL(is_on_host, (double) )
{
  nt2::table<T,nt2::pinned_> B = nt2::ones(nt2::of_size(5,5), nt2::meta::as_<T>());
  nt2::table<T,nt2::device_> A = B;
  nt2::table<T> result = nt2::two(5);

  cublasDscal( A.size() , 2.0 , A.data() , 1);

  auto out_pinned = nt2::to_host<nt2::pinned_>(A);
  auto out_host = nt2::to_host(A);

  NT2_TEST_EQUAL( (std::is_same<typename decltype(out_pinned)::allocator_type,nt2::memory::cuda_pinned_<T> >::value) , true  );
  NT2_TEST_EQUAL( nt2::meta::is_on_host<decltype(out_pinned)>::value , true );
  NT2_TEST_EQUAL( (std::is_same<typename decltype(out_host)::allocator_type,nt2::memory::cuda_pinned_<T> >::value) , false  );
  NT2_TEST_EQUAL( nt2::meta::is_on_host<decltype(out_host)>::value , true );
  NT2_TEST_EQUAL(out_pinned , result );
  NT2_TEST_EQUAL(out_host , result );
}


NT2_TEST_CASE_TPL(is_on_host_inout, (double) )
{
  nt2::table<T,nt2::pinned_> A = nt2::ones(nt2::of_size(5,5), nt2::meta::as_<T>());
  nt2::table<T> result = nt2::two(5);

  auto A_device = nt2::to_device(A);

  cublasDscal( A.size() , 2.0 , A.data() , 1);

  nt2::to_host(A_device,A);

  NT2_TEST_EQUAL( A , result );

}

NT2_TEST_CASE_TPL(is_on_device, (double) )
{
  nt2::table<T,nt2::pinned_> A = nt2::ones(nt2::of_size(5,5), nt2::meta::as_<T>());
  nt2::table<T> result = nt2::two(5);

  auto A_device = nt2::to_device(A);

  cublasDscal( A_device.size() , 2.0 , A_device.data() , 1);

  auto out_pinned = nt2::to_host<nt2::pinned_>(A_device);

  NT2_TEST_EQUAL(out_pinned , result );
  NT2_TEST_EQUAL(A , result );
}
