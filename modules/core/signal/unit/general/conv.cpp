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
NT2_TEST_CASE( full_conv_static )
{
  using nt2::_;
  using nt2::conv;
  using nt2::of_size;

  nt2::table<float, nt2::of_size_<1,15> > u(_(1.f,15.f));
  nt2::table<float, nt2::of_size_<1,6> > v(_(2.f,2,12.f));
  nt2::table<float>res;

  nt2::table<float, nt2::of_size_<1,20> > ref = nt2::cons<float>( of_size(1,20)
                                          , 2.f,8.f,20.f,40.f,70.f,112.f,154.f
                                          , 196.f,238.f,280.f,322.f,364.f,406.f
                                          , 448.f,490.f,500.f,476.f,416.f,318.f
                                          , 180.f);


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
NT2_TEST_CASE( same_conv_static )
{
  using nt2::_;
  using nt2::conv;
  using nt2::same_;
  using nt2::of_size;

  nt2::table<float, nt2::of_size_<1,15> > u(_(1.f,15.f));
  nt2::table<float, nt2::of_size_<1,6> > v(_(2.f,2,12.f));
  nt2::table<float, nt2::of_size_<1,15> > ref = nt2::cons<float>( of_size(1,15)
                                          , 40.f,  70.f, 112.f, 154.f, 196.f
                                          , 238.f, 280.f, 322.f, 364.f, 406.f
                                          , 448.f, 490.f, 500.f, 476.f, 416.f
                                          );
  nt2::table<float, nt2::of_size_<1,6> > rref = nt2::cons<float>( of_size(1,6)
                                            , 196.f, 238.f, 280.f
                                            , 322.f, 364.f, 406.f
                                            );
  nt2::table<float> res;

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

NT2_TEST_CASE_TPL( valid_conv_static , NT2_REAL_TYPES )
{
  using nt2::_;
  using nt2::conv;
  using nt2::isempty;
  using nt2::valid_;
  using nt2::of_size;

  nt2::table<T, nt2::of_size_<1,15> > u(_(T(1),T(15)));
  nt2::table<T, nt2::of_size_<1,6> > v(_(T(2),T(2),T(12)));
  nt2::table<T, nt2::of_size_<1,10> > ref = nt2::cons<T>( of_size(1,10)
                                          , T(112),T(154),T(196),T(238),T(280)
                                          , T(322),T(364),T(406),T(448),T(490)
                                          );
  nt2::table<T> res;

  res = conv(u,v,valid_);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u,valid_);
  NT2_TEST(isempty(res));

  res = conv(v,v,valid_);
  NT2_TEST_ULP_EQUAL(res,T(224),1);

  //size of 7
  nt2::table<T , nt2::of_size_<1,15> > u2(_(T(1),T(15)));
  nt2::table<T , nt2::of_size_<1,7> > v2(_(T(2),T(2),T(14)));
  nt2::table<T , nt2::of_size_<1,9> > ref2 = nt2::cons<T>( of_size(1,9)
                                          , T(168),T(224),T(280),T(336),T(392)
                                          , T(448),T(504),T(560),T(616)
                                          );
  nt2::table<T> res2;

  res2 = conv(u2,v2,valid_);
  NT2_TEST_EQUAL(res2.extent(),ref2.extent());
  NT2_TEST_ULP_EQUAL(res2,ref2,1);

  res2 = conv(v2,u2,valid_);
  NT2_TEST(isempty(res2));

  res2 = conv(v2,v2,valid_);
  NT2_TEST_ULP_EQUAL(res2,T(336),1);

  //FOR simd size of 6
  nt2::table<T, nt2::of_size_<1,100> > u3(_(T(1),T(100)));
  nt2::table<T, nt2::of_size_<1,6> > v3(_(T(2),T(2),T(12)));
  nt2::table<T, nt2::of_size_<1,95> > ref3(_(T(112) , T(42) , T(4060) ));
  nt2::table<T> res3;
  res3 = conv(u3,v3,valid_);
  NT2_TEST_EQUAL(res3.extent(),ref3.extent());
  NT2_TEST_ULP_EQUAL(res3,ref3,1);

  res3 = conv(v3,u3,valid_);
  NT2_TEST(isempty(res3));

  //size of 7

  nt2::table<T, nt2::of_size_<1,100> > u4(_(T(1),T(100)));
  nt2::table<T, nt2::of_size_<1,7> > v4(_(T(2),T(2),T(14)));
  nt2::table<T, nt2::of_size_<1,94> > ref4(_(T(168) , T(56) , T(5376) ));
  nt2::table<T> res4;

  res4 = conv(u4,v4,valid_);
  NT2_TEST_EQUAL(res4.extent(),ref4.extent());
  NT2_TEST_ULP_EQUAL(res4,ref4,1);

  res4 = conv(v4,u4,valid_);
  NT2_TEST(isempty(res4));

  //size of 12

  nt2::table<T, nt2::of_size_<1,100> > u5(_(T(1),T(100)));
  nt2::table<T, nt2::of_size_<1,12> > v5(_(T(2),T(2),T(24)));
  nt2::table<T, nt2::of_size_<1,89> > ref5(_(T(728) , T(156) , T(14456) ));
  nt2::table<T> res5;

  res5 = conv(u5,v5,valid_);
  NT2_TEST_EQUAL(res5.extent(),ref5.extent());
  NT2_TEST_ULP_EQUAL(res5,ref5,1);

  res5 = conv(v5,u5,valid_);
  NT2_TEST(isempty(res5));


}


// DYNAMIC

NT2_TEST_CASE( full_conv )
{
  using nt2::_;
  using nt2::conv;
  using nt2::of_size;

  nt2::table<float> u(_(1.f,15.f));
  nt2::table<float> v(_(2.f,2,12.f));
  nt2::table<float>res;

  nt2::table<float> ref = nt2::cons<float>( of_size(1,20)
                                          , 2.f,8.f,20.f,40.f,70.f,112.f,154.f
                                          , 196.f,238.f,280.f,322.f,364.f,406.f
                                          , 448.f,490.f,500.f,476.f,416.f,318.f
                                          , 180.f);


  res = conv(u,v);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  nt2::table<float > sv = cons( of_size(1,6), 2.f,4.f,6.f,8.f,10.f,12.f);
  res = conv(u,sv);
  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE( same_conv )
{
  using nt2::_;
  using nt2::conv;
  using nt2::same_;
  using nt2::of_size;

  nt2::table<float> u(_(1.f,15.f));
  nt2::table<float> v(_(2.f,2,12.f));
  nt2::table<float> ref = nt2::cons<float>( of_size(1,15)
                                          , 40.f,  70.f, 112.f, 154.f, 196.f
                                          , 238.f, 280.f, 322.f, 364.f, 406.f
                                          , 448.f, 490.f, 500.f, 476.f, 416.f
                                          );
  nt2::table<float> rref = nt2::cons<float>( of_size(1,6)
                                            , 196.f, 238.f, 280.f
                                            , 322.f, 364.f, 406.f
                                            );
  nt2::table<float> res;

  res = conv(u,v,same_);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u,same_);
  NT2_TEST_EQUAL(res.extent(),rref.extent());
  NT2_TEST_ULP_EQUAL(res,rref,1);

  nt2::table<float > sv = cons( of_size(1,6), 2.f,4.f,6.f,8.f,10.f,12.f);
  res = conv(u,sv,same_);
  NT2_TEST_ULP_EQUAL(res,ref,1);



  //size of 7

}

NT2_TEST_CASE_TPL( valid_conv, NT2_REAL_TYPES )
{
  using nt2::_;
  using nt2::conv;
  using nt2::isempty;
  using nt2::valid_;
  using nt2::of_size;

  nt2::table<T> u(_(T(1),T(15)));
  nt2::table<T> v(_(T(2),T(2),T(12)));
  nt2::table<T> ref = nt2::cons<T>( of_size(1,10)
                                          , T(112),T(154),T(196),T(238),T(280)
                                          , T(322),T(364),T(406),T(448),T(490)
                                          );
  nt2::table<T> res;

  res = conv(u,v,valid_);
  NT2_TEST_EQUAL(res.extent(),ref.extent());
  NT2_TEST_ULP_EQUAL(res,ref,1);

  res = conv(v,u,valid_);
  NT2_TEST(isempty(res));

  res = conv(v,v,valid_);
  NT2_TEST_ULP_EQUAL(res,T(224),1);

  //size of 7
  nt2::table<T> u2(_(T(1),T(15)));
  nt2::table<T> v2(_(T(2),T(2),T(14)));
  nt2::table<T> ref2 = nt2::cons<T>( of_size(1,9)
                                          , T(168),T(224),T(280),T(336),T(392)
                                          , T(448),T(504),T(560),T(616)
                                          );
  nt2::table<T> res2;

  res2 = conv(u2,v2,valid_);
  NT2_TEST_EQUAL(res2.extent(),ref2.extent());
  NT2_TEST_ULP_EQUAL(res2,ref2,1);

  res2 = conv(v2,u2,valid_);
  NT2_TEST(isempty(res2));

  res2 = conv(v2,v2,valid_);
  NT2_TEST_ULP_EQUAL(res2,T(336),1);

  //FOR simd size of 6
  nt2::table<T> u3(_(T(1),T(100)));
  nt2::table<T> v3(_(T(2),T(2),T(12)));
  nt2::table<T> ref3(_(T(112) , T(42) , T(4060) ));
  nt2::table<T> res3;
  res3 = conv(u3,v3,valid_);
  NT2_TEST_EQUAL(res3.extent(),ref3.extent());
  NT2_TEST_ULP_EQUAL(res3,ref3,1);

  res3 = conv(v3,u3,valid_);
  NT2_TEST(isempty(res3));

  //size of 7

  nt2::table<T> u4(_(T(1),T(100)));
  nt2::table<T> v4(_(T(2),T(2),T(14)));
  nt2::table<T> ref4(_(T(168) , T(56) , T(5376) ));
  nt2::table<T> res4;

  res4 = conv(u4,v4,valid_);
  NT2_TEST_EQUAL(res4.extent(),ref4.extent());
  NT2_TEST_ULP_EQUAL(res4,ref4,1);

  res4 = conv(v4,u4,valid_);
  NT2_TEST(isempty(res4));

  //size of 12

  nt2::table<T> u5(_(T(1),T(100)));
  nt2::table<T> v5(_(T(2),T(2),T(24)));
  nt2::table<T> ref5(_(T(728) , T(156) , T(14456) ));
  nt2::table<T> res5;

  res5 = conv(u5,v5,valid_);
  NT2_TEST_EQUAL(res5.extent(),ref5.extent());
  NT2_TEST_ULP_EQUAL(res5,ref5,1);

  res5 = conv(v5,u5,valid_);
  NT2_TEST(isempty(res5));

}
