//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/dispatch/meta/is_homogeneous.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/placeholders.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>

NT2_TEST_CASE(base_type_are_homogeneous)
{
  using boost::mpl::_;
  using boost::dispatch::meta::is_homogeneous;

  NT2_TEST( (is_homogeneous<float>::value) );
  NT2_TEST( (is_homogeneous<int**>::value) );
  NT2_TEST( (is_homogeneous<char[7]>::value) );
}

NT2_TEST_CASE(homogeneous_sequence)
{
  using boost::mpl::_;
  using boost::dispatch::meta::is_homogeneous;

  using htype  = boost::fusion::vector<int,int,int,int>;
  using atype  = std::array<int,8>;
  using btype  = boost::array<int,8>;
  using nhtype = boost::fusion::vector<int,void*,float>;

  NT2_TEST( (is_homogeneous<htype>::value) );
  NT2_TEST( (is_homogeneous<atype>::value) );
  NT2_TEST( (is_homogeneous<btype>::value) );
  NT2_TEST( !(is_homogeneous<nhtype>::value) );
}
