//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/include/functions/trsm.hpp>
#include <nt2/include/functions/linsolve.hpp>
#include <nt2/include/functions/rand.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/mtimes.hpp>

#include <nt2/table.hpp>
#include <nt2/include/functions/lu.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <iostream>
#include <nt2/sdk/meta/type_id.hpp>

 NT2_TEST_CASE_TPL(trsm_nt2_device, NT2_REAL_TYPES )
 {
   using nt2::_;

   typedef nt2::table<T>                       t_t;
   typedef nt2::table<T,nt2::device_>         device_t;

   t_t a = nt2::cons<T>(nt2::of_size(3,3),2,1,1,1,1,1,1,1,2);
   t_t b = nt2::cons<T>(nt2::of_size(3,1),1,2,5);
   t_t x = nt2::cons<T>(nt2::of_size(3,1),-1,0,3);

   t_t y(b),p;

   nt2::table<T, nt2::lower_triangular_> l;
   nt2::table<T, nt2::upper_triangular_> u;

   nt2::tie(l,u,p) = nt2::lu(a);

   y = nt2::mtimes(p,b);

   nt2::table<T,nt2::settings(nt2::lower_triangular_ , nt2::device_)>    d_l = l;
   nt2::table<T,nt2::settings(nt2::upper_triangular_ , nt2::device_)>    d_u = u;

   device_t d_y = y;

   nt2::tie(d_y) = nt2::linsolve(d_l,d_y);
   nt2::tie(d_y) = nt2::linsolve(d_u,d_y);

   y = d_y;

   NT2_TEST_EQUAL( y, x);
 }

 NT2_TEST_CASE_TPL(trsm_nt2_device_complex, NT2_REAL_TYPES )
 {
   using nt2::_;

   using cT = std::complex<T>;

   typedef nt2::table<cT>                       t_t;
   typedef nt2::table<T>                        t_p;
   typedef nt2::table<cT,nt2::device_>          device_t;

   t_t a = nt2::cons<cT>(nt2::of_size(3,3)
                    ,cT(2,0),cT(1,0),cT(1,0)
                    ,cT(1,0),cT(1,0),cT(1,0)
                    ,cT(1,0),cT(1,0),cT(2,0)
                    );
   t_t b = nt2::cons<cT>(nt2::of_size(3,1)
                    ,cT(1,0),cT(2,0),cT(5,0));

   t_t x = nt2::cons<cT>(nt2::of_size(3,1),cT(-1,0),cT(0,0),cT(3,0));

   t_t y(b);
   t_p p;

   nt2::table<cT, nt2::lower_triangular_> l;
   nt2::table<cT, nt2::upper_triangular_> u;

   nt2::tie(l,u,p) = nt2::lu(a);

   y = nt2::mtimes(p,b);

   nt2::table<cT,nt2::settings(nt2::lower_triangular_ , nt2::device_)>    d_l = l;
   nt2::table<cT,nt2::settings(nt2::upper_triangular_ , nt2::device_)>    d_u = u;

   device_t d_y = y;

   nt2::tie(d_y) = nt2::linsolve(d_l,d_y);
   nt2::tie(d_y) = nt2::linsolve(d_u,d_y);

   y = d_y;

   NT2_TEST_EQUAL( y, x);
 }


NT2_TEST_CASE_TPL(trsm_device, NT2_REAL_TYPES )
{
  using nt2::_;

  typedef nt2::table<T>                       t_t;
  typedef nt2::table<T, nt2::device_>         device_t;

  t_t a = nt2::cons<T>(nt2::of_size(3,3),2,1,1,1,1,1,1,1,2);
  t_t b = nt2::cons<T>(nt2::of_size(3,1),1,2,5);
  t_t x = nt2::cons<T>(nt2::of_size(3,1),-1,0,3);

  t_t y(b);

  t_t l,u,p;

  nt2::tie(l,u,p) = nt2::lu(a);

  char lside = 'l';
  char uplo = 'u';
  char lplo = 'l';
  char diag = 'n';
  char notrans= 'n';

  y = nt2::mtimes(p,b);

  device_t d_l = l;
  device_t d_u = u;
  device_t d_y = y;

  nt2::trsm(lside,lplo,notrans,diag,boost::proto::value(d_l),boost::proto::value(d_y));

  nt2::trsm(lside,uplo,notrans,diag,boost::proto::value(d_u),boost::proto::value(d_y));

  t_t result = d_y ;

  NT2_TEST_EQUAL( result, x);
}


NT2_TEST_CASE_TPL(trsm_device_complex, NT2_REAL_TYPES )
{
  using nt2::_;

   using cT = std::complex<T>;

   typedef nt2::table<cT>                       t_t;
   typedef nt2::table<T>                        t_p;
   typedef nt2::table<cT,nt2::device_>          device_t;

   t_t a = nt2::cons<cT>(nt2::of_size(3,3)
                    ,cT(2,0),cT(1,0),cT(1,0)
                    ,cT(1,0),cT(1,0),cT(1,0)
                    ,cT(1,0),cT(1,0),cT(2,0)
                    );
   t_t b = nt2::cons<cT>(nt2::of_size(3,1)
                    ,cT(1,0),cT(2,0),cT(5,0));

   t_t x = nt2::cons<cT>(nt2::of_size(3,1),cT(-1,0),cT(0,0),cT(3,0));

   t_t y(b),l,u;
   t_p p;

  nt2::tie(l,u,p) = nt2::lu(a);

  char lside = 'l';
  char uplo = 'u';
  char lplo = 'l';
  char diag = 'n';
  char notrans= 'n';

  y = nt2::mtimes(p,b);

  device_t d_l = l;
  device_t d_u = u;
  device_t d_y = y;

  nt2::trsm(lside,lplo,notrans,diag,boost::proto::value(d_l),boost::proto::value(d_y));

  nt2::trsm(lside,uplo,notrans,diag,boost::proto::value(d_u),boost::proto::value(d_y));

  t_t result = d_y ;

  NT2_TEST_EQUAL( result, x);
}


NT2_TEST_CASE_TPL(trsm_device_host_device_alloc, NT2_REAL_TYPES )
{
  using nt2::_;

  typedef nt2::table<T>                       t_t;
  typedef nt2::table<T, nt2::device_>         device_t;

  t_t a = nt2::cons<T>(nt2::of_size(3,3),2,1,1,1,1,1,1,1,2);
  t_t b = nt2::cons<T>(nt2::of_size(3,1),1,2,5);
  t_t x = nt2::cons<T>(nt2::of_size(3,1),-1,0,3);

  t_t y(b);

  t_t l,u,p;

  nt2::tie(l,u,p) = nt2::lu(a);

  char lside = 'l';
  char uplo = 'u';
  char lplo = 'l';
  char diag = 'n';
  char notrans= 'n';

  y = nt2::mtimes(p,b);

  device_t d_l = l;
  device_t d_u = u;

  nt2::trsm(lside,lplo,notrans,diag,boost::proto::value(d_l),boost::proto::value(y));

  nt2::trsm(lside,uplo,notrans,diag,boost::proto::value(d_u),boost::proto::value(y));

  NT2_TEST_EQUAL( y, x);
}


NT2_TEST_CASE_TPL(trsm_device_host_alloc, NT2_REAL_TYPES )
{
  using nt2::_;

  typedef nt2::table<T>                       t_t;

  t_t a = nt2::cons<T>(nt2::of_size(3,3),2,1,1,1,1,1,1,1,2);
  t_t b = nt2::cons<T>(nt2::of_size(3,1),1,2,5);
  t_t x = nt2::cons<T>(nt2::of_size(3,1),-1,0,3);

  t_t y(b);

  t_t l,u,p;

  nt2::tie(l,u,p) = nt2::lu(a);

  char lside = 'l';
  char uplo = 'u';
  char lplo = 'l';
  char diag = 'n';
  char notrans= 'n';

  y = nt2::mtimes(p,b);

  nt2::trsm(lside,lplo,notrans,diag,boost::proto::value(l),boost::proto::value(y));

  nt2::trsm(lside,uplo,notrans,diag,boost::proto::value(u),boost::proto::value(y));

  NT2_TEST_EQUAL( y, x);
}
