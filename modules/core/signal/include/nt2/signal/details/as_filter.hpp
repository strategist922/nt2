//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_DETAILS_AS_FILTER_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_AS_FILTER_HPP_INCLUDED

#include <nt2/include/functions/run.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>
#include <boost/simd/include/functions/simd/splat.hpp>
#include <boost/simd/include/functions/simd/plus.hpp>
#include <boost/simd/include/functions/simd/multiplies.hpp>
#include <boost/simd/sdk/meta/scalar_of.hpp>
#include <boost/dispatch/meta/as.hpp>

namespace nt2
{
  namespace details
  {
    // Adapt a table to the ConvolutionOperator concept
    template<typename Expression>
    struct dynamic_filter
    {
      typedef typename Expression::extent_type                          extent_type;
      typedef typename Expression::value_type                           v_t;
      typedef memory::container<tag::table_, v_t, extent_type>          sema_t;
      typedef typename container::as_terminal<sema_t, Expression>::type f_t;

      typedef typename boost::dispatch::meta
                            ::call<tag::numel_(extent_type const&)>::type size_type;

      BOOST_FORCEINLINE dynamic_filter( Expression const& e ) : filter_(e) {}

      // TODO: C++11 use auto return type instead
      BOOST_FORCEINLINE size_type size()    const
      {
        return numel(filter_.extent());
      }

      // conv perform the inner operation between
      // the correlated and the correlator
      template<typename T>
      BOOST_FORCEINLINE T conv(T const& data, std::size_t index) const
      {
        typedef typename boost::simd::meta::scalar_of<T>::type s_type;
        return data * boost::simd::splat<T>( filter_(index) );
      }

      // reduce aggregate correlation partial result
      template<typename T>
      BOOST_FORCEINLINE T reduce(T const& reduced,T const& elem) const
      {
        return reduced + elem;
      }

      // normalize may post-process the final result
      template<typename T>
      BOOST_FORCEINLINE T normalize(T const& v) const
      {
        return v;
      }

      private:
      f_t filter_;
    };
  }

  template<typename Expression>
  BOOST_FORCEINLINE details::dynamic_filter<Expression>
  as_filter ( Expression const& e )
  {
    return details::dynamic_filter<Expression>(e);
  }
}

#endif
