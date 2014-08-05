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
#include <boost/simd/include/functions/simd/splat.hpp>
#include <boost/simd/include/functions/simd/plus.hpp>
#include <boost/simd/include/functions/simd/multiplies.hpp>
#include <boost/simd/sdk/meta/scalar_of.hpp>
#include <boost/dispatch/meta/as.hpp>

namespace nt2
{
  namespace details
  {
    template<typename Expression>
    struct dynamic_filter
    {
      dynamic_filter( Expression const& e)
                    : filter_(e), size_(numel(e))
      {}

      std::size_t size() const { return size_; }

      template<typename T> BOOST_FORCEINLINE T conv(T const& data, std::size_t index) const
      {
        typedef typename boost::simd::meta::scalar_of<T>::type s_type;
        return data * boost::simd::splat<T>( nt2::run(filter_,index,meta::as_<s_type>()) );
      }

      template<typename T> BOOST_FORCEINLINE T reduce(T const& reduced,T const& elem) const
      {
        return reduced + elem;
      }

      private:
      Expression const& filter_;
      std::size_t       size_;
    };
  }

  template<typename Expression>
  details::dynamic_filter<Expression> as_filter(Expression const& e)
  {
    return details::dynamic_filter<Expression>(e);
  }
}

#endif
