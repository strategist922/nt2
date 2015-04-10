//==============================================================================
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <nt2/include/functions/extent.hpp>
#include <nt2/include/functions/of_size.hpp>
#include <boost/mpl/vector.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>

////////////////////////////////////////////////////////////////////////////////
// extent of arithmetic types
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL( fundamental_extent, NT2_TYPES )
{
  using nt2::extent;
  using boost::mpl::_;

  NT2_TEST_EQUAL    ( extent(T(4)), nt2::of_size(1)  );
  NT2_TEST_EXPR_TYPE( extent(T(4)), _ , (nt2::_0D)   );
}

////////////////////////////////////////////////////////////////////////////////
// extent of container
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE( container_extent )
{
  using nt2::extent;
  using nt2::tag::table_;
  using nt2::of_size;

  typedef nt2::memory::container<table_,float,nt2::settings()> container_t;

  container_t t0;
  container_t t1( of_size(2,1,1,1) );
  container_t t2( of_size(2,3,1,1) );
  container_t t3( of_size(2,3,4,1) );
  container_t t4( of_size(2,3,4,5) );

  NT2_TEST_EQUAL( extent(t0), of_size(0)        );
  NT2_TEST_EQUAL( extent(t1), of_size(2,1,1,1)  );
  NT2_TEST_EQUAL( extent(t2), of_size(2,3,1,1)  );
  NT2_TEST_EQUAL( extent(t3), of_size(2,3,4,1)  );
  NT2_TEST_EQUAL( extent(t4), of_size(2,3,4,5)  );
}

////////////////////////////////////////////////////////////////////////////////
// extent of table
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE( table_extent )
{
  using nt2::extent;
  using nt2::of_size;
  using nt2::table;

  typedef table<float> container_t;

  container_t t0;
  container_t t1( of_size(2,1,1,1) );
  container_t t2( of_size(2,3,1,1) );
  container_t t3( of_size(2,3,4,1) );
  container_t t4( of_size(2,3,4,5) );

  NT2_TEST_EQUAL( extent(t0), of_size(0) );
  NT2_TEST_EQUAL( extent(t1), of_size(2,1,1,1)  );
  NT2_TEST_EQUAL( extent(t2), of_size(2,3,1,1)  );
  NT2_TEST_EQUAL( extent(t3), of_size(2,3,4,1)  );
  NT2_TEST_EQUAL( extent(t4), of_size(2,3,4,5)  );
}

////////////////////////////////////////////////////////////////////////////////
// extent of unary elementwise expression
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE( unary_elementwise_extent )
{
  using nt2::extent;
  using nt2::of_size;
  using nt2::table;

  typedef table<float> container_t;

  container_t t0;
  container_t t1( of_size(2,1,1,1) );
  container_t t2( of_size(2,3,1,1) );
  container_t t3( of_size(2,3,4,1) );
  container_t t4( of_size(2,3,4,5) );

  NT2_TEST_EQUAL( extent(-t0), of_size(0)        );
  NT2_TEST_EQUAL( extent(-t1), of_size(2,1,1,1)  );
  NT2_TEST_EQUAL( extent(-t2), of_size(2,3,1,1)  );
  NT2_TEST_EQUAL( extent(-t3), of_size(2,3,4,1)  );
  NT2_TEST_EQUAL( extent(-t4), of_size(2,3,4,5)  );
}

////////////////////////////////////////////////////////////////////////////////
// extent of binary elementwise expression
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE( binary_elementwise_extent )
{
  using nt2::extent;
  using nt2::of_size;
  using nt2::table;

  typedef table<float> container_t;

  container_t t0;
  container_t t1( of_size(2,1,1,1) );
  table<float, nt2::of_size_<2,1> > s1;
  container_t t2( of_size(2,3,1,1) );
  container_t t3( of_size(2,3,4,1) );
  container_t t4( of_size(2,3,4,5) );

  NT2_TEST_EQUAL( extent(1.f+t0), of_size(0)        );
//  NT2_TEST_EQUAL( extent(t0-1.f), of_size(0)        );
  //NT2_TEST_EQUAL( extent(t0+t0), of_size(0)        );

/*  NT2_TEST_EQUAL( extent(1.f+t1), of_size(2,1,1,1)     );
  NT2_TEST_EQUAL( extent(t1-1.f), of_size(2,1,1,1)        );
  NT2_TEST_EQUAL( extent(1.f+s1), of_size(2,1,1,1)        );
  NT2_TEST_EQUAL( extent(s1-1.f), of_size(2,1,1,1)        );*/
/*  NT2_TEST_EQUAL( extent(t1+t1), of_size(2,1,1,1)        );
  NT2_TEST_EQUAL( extent(t1+s1), of_size(2,1,1,1)        );
  NT2_TEST_EQUAL( extent(s1+s1), of_size(2,1,1,1)        );
  NT2_TEST_EQUAL( extent(s1+t1), of_size(2,1,1,1)        );*/

/*  NT2_TEST_EQUAL( extent(1.f+t1), of_size(2,1,1,1)  );
  NT2_TEST_EQUAL( extent(1.f+t2), of_size(2,3,1,1)  );
  NT2_TEST_EQUAL( extent(1.f+t3), of_size(2,3,4,1)  );
  NT2_TEST_EQUAL( extent(1.f+t4), of_size(2,3,4,5)  );
*/}

/*
////////////////////////////////////////////////////////////////////////////////
// extent of table expression
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE( expression_extent )
{
  using nt2::extent;
  using nt2::of_size;
  using nt2::table;

  table<float> t0;
  table<float> t1( of_size(2) );
  table<float> t2( of_size(2,2) );
  table<float> t3( of_size(2,2,2) );
  table<float> t4( of_size(2,2,2,2) );

  NT2_TEST_EQUAL( extent(-t0), 0U   );
  NT2_TEST_EQUAL( extent(t1*t1), 2U   );
  NT2_TEST_EQUAL( extent(t2-t2*t2), 4U   );
  NT2_TEST_EQUAL( extent(t3/t3+t3), 8U   );
  NT2_TEST_EQUAL( extent(t4 * -t4), 16U  );
}

NT2_TEST_CASE( static_extent )
{
  using nt2::extent;
  using nt2::of_size_;
  using nt2::table;
  using boost::mpl::_;

  float a;
  of_size_<2, 3> b;
  table<float, of_size_<2, 3> > c;
  boost::mpl::vector_c<nt2::int16_t,2,3,4> mv;

  NT2_TEST_EXPR_TYPE( extent(a)
                    , _
                    , boost::mpl::size_t<1>
                    );

  NT2_TEST_EXPR_TYPE( extent(b)
                    , _
                    , ( boost::mpl::integral_c<std::size_t, 6> )
                    );

  NT2_TEST_EXPR_TYPE( extent(c)
                    , _
                    , ( boost::mpl::integral_c<std::size_t, 6> )
                    );

  NT2_TEST_EXPR_TYPE( extent(mv)
                    , _
                    , ( boost::mpl::integral_c<nt2::int16_t, 24> )
                    );
}
*/
