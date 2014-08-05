//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_COMMON_FILTER_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_FILTER_HPP_INCLUDED

#ifndef BOOST_SIMD_NO_SIMD

#include <nt2/signal/functions/filter.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/include/functions/run.hpp>
#include <boost/simd/sdk/simd/meta/is_vectorizable.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/sdk/meta/scalar_of.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <boost/dispatch/meta/as.hpp>

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION_IF ( nt2::tag::filter_, boost::simd::tag::simd_
                                , (A0)(N0)(A1)(N1)
                                , (boost::simd::meta::is_vectorizable< typename A0::value_type
                                                                     , BOOST_SIMD_DEFAULT_EXTENSION
                                                                     >
                                  )
                                , ((node_< A0, nt2::tag::filter_
                                         , N0, nt2::container::domain
                                         >
                                  ))
                                  ((node_< A1, nt2::tag::tie_
                                         , N1, nt2::container::domain
                                         >
                                  ))
                                )
  {
    typedef void result_type;

    typedef typename boost::proto::result_of::child_c<A0&,0>::type child0_t;
    typedef typename boost::proto::result_of::value<child0_t>::type filter_t;
    typedef typename boost::proto::result_of::child_c<A0&,2>::value_type child2_t;
    typedef typename child2_t::value_type real_type;

    BOOST_FORCEINLINE result_type operator()( A0 const& a0, A1 const& a1 ) const
    {
      eval(a0, a1, N0());
    }

    result_type eval(A0 const& a0, A1 const& a1, boost::mpl::long_<4> const&) const
    {

    }

    result_type eval(A0 const& a0, A1 const& a1, boost::mpl::long_<3> const&) const
    {
      typedef boost::simd::native<real_type,BOOST_SIMD_DEFAULT_EXTENSION> n_t;

      filter_t const& f = boost::proto::value(boost::proto::child_c<0>(a0));

      static const std::size_t cd = boost::simd::meta::cardinal_of<n_t>::value;
      std::size_t ds = boost::proto::child_c<2>(a0).size();
      std::size_t fs = f.size()-1;
      std::size_t ms = std::min(f.size(),ds);
      std::size_t ls = ds>cd ? ds-cd : 0;
      std::size_t ii = 0;

      boost::proto::child_c<0>(a1).resize(boost::proto::child_c<2>(a0).extent());

      for (;ii<ms;ii++)
      {
        real_type res = f.conv(nt2::run(boost::proto::child_c<2>(a0),0,meta::as_<real_type>()),ii);

        for (std::size_t jj=1;jj<=ii;jj++)
        {
          real_type dd = nt2::run(boost::proto::child_c<2>(a0),jj,meta::as_<real_type>());
          res = f.reduce(res,f.conv(dd,ii-jj));
        }
        nt2::run(boost::proto::child_c<0>(a1),ii,res);
      }

      for (;ii<ls;ii+=cd)
      {
        n_t dd = nt2::run(boost::proto::child_c<2>(a0),ii-fs,meta::as_<n_t>());
        n_t res = f.conv(dd,fs);
        for (std::size_t jj=1;jj<=fs;jj++)
        {
          dd = nt2::run(boost::proto::child_c<2>(a0),ii-fs+jj,meta::as_<n_t>());
          res = f.reduce(res,f.conv(dd,fs-jj));
        }
        nt2::run(boost::proto::child_c<0>(a1),ii,res);
      }

      for (;ii<ds;ii++)
      {
        real_type res = f.conv(nt2::run(boost::proto::child_c<2>(a0),ii-fs,meta::as_<real_type>()),fs);
        for (std::size_t jj=1;jj<=fs;jj++)
        {
          real_type dd = nt2::run(boost::proto::child_c<2>(a0),ii-fs+jj,meta::as_<real_type>());
          res = f.reduce(res,f.conv(dd,fs-jj));
        }
        nt2::run(boost::proto::child_c<0>(a1),ii,res);
      }
    }
  };
} }

#endif
#endif
