//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <nt2/signal/include/functions/conv.hpp>
#include <nt2/include/functions/isempty.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/zeros.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>

NT2_TEST_CASE( conv_size )
{
  using nt2::conv;
  using nt2::of_size;
  using nt2::full_;
  using nt2::same_;
  using nt2::valid_;

  nt2::table<float> u(of_size(13))
                  , v(of_size(3))
                  , tu(of_size(1,13))
                  , tv(of_size(1,3));

  NT2_TEST_EQUAL( conv(u,v).extent()        , of_size(15,1));
  NT2_TEST_EQUAL( conv(u,v,full_ ).extent() , of_size(15,1));
  NT2_TEST_EQUAL( conv(u,v,same_ ).extent() , u.extent());
  NT2_TEST_EQUAL( conv(u,v,valid_).extent() , of_size(11,1) );

  NT2_TEST_EQUAL( conv(v,u).extent()        , of_size(15,1));
  NT2_TEST_EQUAL( conv(v,u,full_ ).extent() , of_size(15,1));
  NT2_TEST_EQUAL( conv(v,u,same_ ).extent() , v.extent());
  NT2_TEST_EQUAL( conv(v,u,valid_).extent() , of_size(0,1) );

  NT2_TEST_EQUAL( conv(tu,v).extent()        , of_size(1,15) );
  NT2_TEST_EQUAL( conv(tu,v,full_ ).extent() , of_size(1,15) );
  NT2_TEST_EQUAL( conv(tu,v,same_ ).extent() , tu.extent()   );
  NT2_TEST_EQUAL( conv(tu,v,valid_).extent() , of_size(1,11) );

  NT2_TEST_EQUAL( conv(tv,u).extent()        , of_size(1,15) );
  NT2_TEST_EQUAL( conv(tv,u,full_ ).extent() , of_size(1,15) );
  NT2_TEST_EQUAL( conv(tv,u,same_ ).extent() , tv.extent()   );
  NT2_TEST_EQUAL( conv(tv,u,valid_).extent() , of_size(1,0)  );
}

NT2_TEST_CASE( conv_check )
{
  using nt2::conv;
  using nt2::of_size;

  nt2::table<float> u(of_size(3,3))
                  , v(of_size(3));

  NT2_TEST_ASSERT( (conv(u,v)) );
  NT2_TEST_ASSERT( (conv(v,u)) );
}

NT2_TEST_CASE( full_conv )
{
  using nt2::_;
  using nt2::conv;
  using nt2::of_size;

  nt2::table<float> u(_(1.f,15.f))
                  , v(_(2.f,2,12.f))
                  , ref = nt2::cons<float>( of_size(1,20)
                                          , 2.f,8.f,20.f,40.f,70.f,112.f,154.f
                                          , 196.f,238.f,280.f,322.f,364.f,406.f
                                          , 448.f,490.f,500.f,476.f,416.f,318.f
                                          , 180.f
                                          )
                  , res;


  res = conv(u,v);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  nt2::table<float, nt2::of_size_<1,6> > sv = cons( of_size(1,6), 2.f,4.f,6.f,8.f,10.f,12.f);
  res = conv(u,sv);
  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE( same_conv )
{
  using nt2::_;
  using nt2::conv;
  using nt2::same_;
  using nt2::of_size;

  nt2::table<float> u(_(1.f,15.f))
                  , v(_(2.f,2,12.f))
                  , ref = nt2::cons<float>( of_size(1,15)
                                          , 40.f,  70.f, 112.f, 154.f, 196.f
                                          , 238.f, 280.f, 322.f, 364.f, 406.f
                                          , 448.f, 490.f, 500.f, 476.f, 416.f
                                          )
                  , rref = nt2::cons<float> ( of_size(1,6)
                                            , 196.f, 238.f, 280.f
                                            , 322.f, 364.f, 406.f
                                            )
                  , res;

  res = conv(u,v,same_);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u,same_);
  NT2_TEST_EQUAL(res.extent(),rref.extent());
  NT2_TEST_ULP_EQUAL(res,rref,1);

  nt2::table<float, nt2::of_size_<1,6> > sv = cons( of_size(1,6), 2.f,4.f,6.f,8.f,10.f,12.f);
  res = conv(u,sv,same_);
  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE( valid_conv )
{
  using nt2::_;
  using nt2::conv;
  using nt2::isempty;
  using nt2::valid_;
  using nt2::of_size;

  nt2::table<float> u(_(1.f,15.f))
                  , v(_(2.f,2,12.f))
                  , ref = nt2::cons<float>( of_size(1,10)
                                          , 112.f,154.f,196.f,238.f,280.f
                                          , 322.f,364.f,406.f,448.f,490.f
                                          )
                  , res;

  res = conv(u,v,valid_);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u,valid_);
  NT2_TEST(isempty(res));

  res = conv(v,v,valid_);
  NT2_TEST_ULP_EQUAL(res,224.f,1);
}
