//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SCALAR_FILTER_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SCALAR_FILTER_HPP_INCLUDED

#ifndef BOOST_SIMD_NO_SIMD

#include <nt2/signal/functions/filter.hpp>

#include <nt2/core/container/table/table.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>
#include <nt2/include/functions/run.hpp>
#include <boost/dispatch/meta/as.hpp>

#include <nt2/signal/details/filter_imp.hpp>

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION ( nt2::tag::filter_, tag::cpu_
                             , (A0)(N0)(A1)(N1)
                             , ((node_<A0, nt2::tag::filter_
                                     , N0, nt2::container::domain
                                       >
                               ))
                               ((node_<A1, nt2::tag::tie_
                                     , N1, nt2::container::domain
                                      >
                               ))
                             )
  {
    typedef void result_type;

    typedef typename boost::proto::result_of::child_c<A0&,0>::value_type child0_t;

    typedef typename boost::proto::result_of::child_c<A0&, 2>::value_type child2_t;

    typedef typename meta::scalar_of<child2_t>::type f_type;
    typedef typename boost::dispatch::meta::as_floating<f_type>::type real_type;

    BOOST_FORCEINLINE result_type operator()( A0 const& a0, A1 const& a1 ) const
    {
      eval(a0, a1, N0());
    }

    result_type eval(A0 const& a0, A1 const& a1, boost::mpl::long_<4> const&) const
    {

    }

    result_type eval(A0 const& a0, A1 const& a1, boost::mpl::long_<3> const&) const
    {
      details::filter<child0_t> f(boost::proto::child_c<0>(a0));

      std::size_t ds = boost::proto::child_c<2>(a0).size();
      std::size_t fs = boost::proto::child_c<0>(a0).size()-1;
      std::size_t ms = std::min(boost::proto::child_c<0>(a0).size(),ds);
      std::size_t ii=0;

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

      for (;ii<ds;ii++)
      {
        real_type dd = nt2::run(boost::proto::child_c<2>(a0),ii-fs,meta::as_<real_type>());
        real_type res = f.conv(dd,fs);
        for (std::size_t jj=1;jj<=fs;jj++)
        {
          dd = nt2::run(boost::proto::child_c<2>(a0),ii-fs+jj,meta::as_<real_type>());
          res = f.reduce(res,f.conv(dd,fs-jj));
        }
        nt2::run(boost::proto::child_c<0>(a1),ii,res);
      }
    }
  };
} }

#endif
#endif
