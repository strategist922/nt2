//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#define NT2_UNIT_MODULE "nt2 odeint toolbox - filter"


#include <nt2/sdk/unit/tests.hpp>
#include <nt2/sdk/unit/module.hpp>

#include <nt2/table.hpp>
#include <nt2/include/functions/filter.hpp>
#include <iostream>

#include <nt2/include/functions/tie.hpp>

#include <nt2/include/functions/linspace.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/divides.hpp>

NT2_TEST_CASE_TPL( filter, (double))//(float))//NT2_REAL_TYPES )
{
  const std::size_t data_size = 35;
  nt2::table<T, nt2::of_size_<1,data_size> > data = /*nt2::cons<T>(nt2::of_size(1,data_size),1.,2.,3.);*/ nt2::linspace<T>(T(1),T(data_size),data_size);//nt2::cons<T>(1.,1.2,1);
  NT2_DISPLAY(data);
  const std::size_t size_filt = 7;

  nt2::table<T, nt2::of_size_<1,size_filt> > filt =nt2::ones(nt2::of_size(1,size_filt),nt2::meta::as_<T>());// / T(size_filt);
  NT2_DISPLAY(filt);
  nt2::details::filter filt_type;
  nt2::table<T, nt2::of_size_<1,data_size> > res;
  nt2::tie(res) = nt2::filter(filt,1,data,filt_type);

  NT2_DISPLAY(res);

  //nt2::filter();
}
