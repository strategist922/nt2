//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_DETAILS_FILTER_IMP_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_FILTER_IMP_HPP_INCLUDED

#include <boost/dispatch/meta/as.hpp>
#include <boost/simd/sdk/meta/scalar_of.hpp>
#include <boost/simd/include/functions/simd/splat.hpp>
#include <nt2/include/functions/run.hpp>
#include <nt2/core/container/table/table.hpp>

namespace nt2 { namespace details
{
  template<typename A>
  struct filter
  {
    filter ( A const& filt_)
           : filtA(filt_)
           { }

    A const& filtA;
    std::size_t size;

    template<typename T>
    BOOST_FORCEINLINE T conv(T const& data, std::size_t index) const
    {
      typedef typename boost::simd::meta::scalar_of<T>::type s_type;
      return data * boost::simd::splat<T>( nt2::run(filtA,index,meta::as_<s_type>()) );
    }

    template<typename T>
    BOOST_FORCEINLINE T reduce(T const& a,T const& b) const
    {
      return a + b;
    }
  };

} }

#endif
