//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/array.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/adapted/array.hpp>

#include <boost/dispatch/meta/fusion.hpp>
#include <boost/dispatch/meta/hierarchy_of.hpp>
#include <boost/dispatch/meta/nth_hierarchy.hpp>
#include <boost/type_traits/is_same.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>

#define M0(z,n,t) ::parent
#define UP(T,N) T BOOST_PP_REPEAT(N,M0,~)

////////////////////////////////////////////////////////////////////////////////
// Test that hierarchy_of is correct for ref, value and const ref
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL(hierarchy_of_ref_cref, BOOST_SIMD_TYPES)
{
  using boost::array;
  using boost::mpl::_;
  using namespace boost::dispatch::meta;

  typedef array<T,7> base;
  typedef typename hierarchy_of<base>::type        hvalue;
  typedef typename hierarchy_of<base const>::type  chvalue;
  typedef typename hierarchy_of<base&>::type       rvalue;
  typedef typename hierarchy_of<base const&>::type crvalue;

  NT2_TEST_EXPR_TYPE( rvalue(),  _, hvalue );
  NT2_TEST_EXPR_TYPE( crvalue(), _, chvalue );
}

////////////////////////////////////////////////////////////////////////////////
// Test that hierarchy_of is correct for random fusion sequence
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE(hierarchy_of_fusion_vector)
{
  using boost::fusion::vector5;
  using boost::is_same;
  using namespace boost::dispatch::meta;

  typedef vector5<void**,double,int*,char&,float> type;
  typedef hierarchy_of< type >::type base;

  NT2_TEST_TYPE_IS( (fusion_sequence_< type,boost::mpl::size_t<5> >), base );
  NT2_TEST_TYPE_IS( (unspecified_< type >), UP(base,1) );
}

////////////////////////////////////////////////////////////////////////////////
// Test that hierarchy_of is correct for array
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE(hierarchy_of_array)
{
  using boost::array;
  using boost::mpl::int_;
  using namespace boost::dispatch::meta;

  using type = array<double,7>;

  // NT2_TEST_TYPE_IS( (array_<generic_<double_< type > >, boost::mpl::size_t<7>  >     ) , UP(base,8) );
  // NT2_TEST_TYPE_IS( (array_<generic_<type64_< type > >, boost::mpl::size_t<7>  >     ) , UP(base,9) );
  // NT2_TEST_TYPE_IS( (array_<generic_<floating_sized_< type > >, boost::mpl::size_t<7>  > ) , UP(base,10) );
  // NT2_TEST_TYPE_IS( (array_<generic_<floating_< type > >, boost::mpl::size_t<7>  >       ) , UP(base,11) );
  // NT2_TEST_TYPE_IS( (array_<generic_<signed_< type > >, boost::mpl::size_t<7>  >     ) , UP(base,12) );
  // NT2_TEST_TYPE_IS( (array_<generic_<arithmetic_< type > >, boost::mpl::size_t<7>  > ) , UP(base,13) );
  // NT2_TEST_TYPE_IS( (array_<generic_<fundamental_< type > >, boost::mpl::size_t<7>  >) , UP(base,14) );
  // NT2_TEST_TYPE_IS( (array_<generic_<unspecified_< type > >, boost::mpl::size_t<7>  >) , UP(base,15) );
  // NT2_TEST_TYPE_IS( (array_<unspecified_< type >, boost::mpl::size_t<7>  >           ) , UP(base,16) );
  // NT2_TEST_TYPE_IS( (fusion_sequence_< type,boost::mpl::size_t<7> >                    ) , UP(base,17) );
  // NT2_TEST_TYPE_IS( (unspecified_< type >                        ) , UP(base,18) );

  NT2_TEST_TYPE_IS( (array_<scalar_<double_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<0>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<type64_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<1>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<floating_sized_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<2>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<floating_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<3>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<signed_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<4>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<arithmetic_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<5>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<fundamental_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<6>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<scalar_<unspecified_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<7>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<double_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<8>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<type64_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<9>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<floating_sized_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<10>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<floating_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<11>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<signed_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<12>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<arithmetic_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<13>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<fundamental_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<14>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<generic_<unspecified_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<15>>::type)
                  );

  NT2_TEST_TYPE_IS( (array_<unspecified_<type>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<16>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<double_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<17>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<type64_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<18>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<floating_sized_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<19>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<floating_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<20>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<signed_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<21>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<arithmetic_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<22>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<fundamental_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<23>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<scalar_<unspecified_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<24>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<double_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<25>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<type64_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<26>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<floating_sized_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<27>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<floating_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<28>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<signed_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<29>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<arithmetic_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<30>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<fundamental_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<31>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<generic_<unspecified_<type>>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<32>>::type)
                  );

  NT2_TEST_TYPE_IS( (homogeneous_<unspecified_<type>, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<33>>::type)
                  );

  NT2_TEST_TYPE_IS( (fusion_sequence_<type, boost::mpl::size_t<7>>)
                  , (nth_hierarchy<type,int_<34>>::type)
                  );

  NT2_TEST_TYPE_IS( (unspecified_<type>)
                  , (nth_hierarchy<type,int_<35>>::type)
                  );
}
