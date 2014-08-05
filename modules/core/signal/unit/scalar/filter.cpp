//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/include/functions/filter.hpp>
#include <nt2/include/functions/linspace.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/table.hpp>

#include <nt2/sdk/unit/tests.hpp>
#include <nt2/sdk/unit/module.hpp>

NT2_TEST_CASE_TPL( small_filter, NT2_TYPES )
{
  const std::size_t data_size = 5;
  nt2::table<T> data = nt2::linspace<T>(T(1),T(data_size),data_size);

  const std::size_t size_filt = 7;
  nt2::table<T> filt =nt2::linspace<T>(T(1),T(size_filt),size_filt);

  nt2::table<T> res;
  nt2::tie(res) = nt2::filter(filt,1,data);

  nt2::table<T> ref = nt2::cons<T>(nt2::of_size(1,data_size),1, 4, 10, 20, 35);
  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE_TPL( same_filter, NT2_REAL_TYPES )
{
  const std::size_t data_size = 7;
  nt2::table<T> data = nt2::linspace<T>(T(1),T(data_size),data_size);

  const std::size_t size_filt = 7;
  nt2::table<T> filt =nt2::linspace<T>(T(1),T(size_filt),size_filt);

  nt2::table<T> res;

  nt2::table<T> ref = nt2::cons<T>(nt2::of_size(1,data_size),1, 4, 10, 20, 35, 56, 84);
  nt2::tie(res) = nt2::filter(filt,1,data);

  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE_TPL( big_filter,NT2_REAL_TYPES )
{
  const std::size_t data_size = 20;
  nt2::table<T> data = nt2::linspace<T>(T(1),T(data_size),data_size);
  const std::size_t size_filt = 5 ;

  nt2::table<T> filt =nt2::linspace<T>(T(1),T(size_filt),size_filt);
  nt2::table<T> res;
  nt2::tie(res) = nt2::filter(filt,1,data);

  nt2::table<T> ref = nt2::cons<T>(nt2::of_size(1,data_size),1,4,10,20,35,50,65,80,95,110,125,140,155,170,185,200,215,230,245,260);

  NT2_TEST_ULP_EQUAL(res,ref,1);
}

NT2_TEST_CASE_TPL( random_data_filter,NT2_REAL_TYPES )
{
  const std::size_t data_size = 20;
  nt2::table<T> data = nt2::cons<T> ( nt2::of_size(1,data_size)
                                    , 0.7788, 0.4235, 0.0908, 0.2665, 0.1537, 0.2810, 0.4401, 0.5271, 0.4574, 0.8754
                                    , 0.2548, 0.2240, 0.6678, 0.8444, 0.3445, 0.7805, 0.6753, 0.0067, 0.6022, 0.3868
                                    );

  const std::size_t size_filt = 5;
  nt2::table<T> filt = nt2::cons<T>(nt2::of_size(1,size_filt), 0.5181, 0.9436, 0.6377, 0.9577, 0.2407);

  nt2::table<T> res;
  nt2::tie(res) = nt2::filter(filt,1,data);

  nt2::table<T> ref = nt2::cons<T>(nt2::of_size(1,data_size)
                                  , 4.034962800000000e-01, 9.542910300000000e-01, 9.432988400000001e-1, 1.239675240000000
                                  , 9.820476399999999e-01, 6.494600800000000e-01, 8.682645100000002e-1, 1.078907610000000
                                  , 1.321111560000000, 1.710399520000000, 1.860459040000000, 1.479651210000000
                                  , 1.668306300000000, 1.665195260000000, 1.676972510000000, 1.961389990000000
                                  , 2.275461720000000, 1.671583930000000, 1.579366750000000, 1.607510750000000
                                  );

  NT2_TEST_ULP_EQUAL(res,ref,1);
}
