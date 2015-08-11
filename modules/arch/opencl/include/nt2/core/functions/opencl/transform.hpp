// ==============================================================================
//         Copyright 2013 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI

//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
// ==============================================================================
#ifndef NT2_CORE_FUNCTIONS_OPENCL_TRANSFORM_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_OPENCL_TRANSFORM_HPP_INCLUDED

#if defined(NT2_HAS_OPENCL)

#include <nt2/core/functions/transform.hpp>
#include <nt2/sdk/opencl/opencl.hpp>
#include <nt2/sdk/external_kernel/external_kernel.hpp>
#include <nt2/core/settings/locality.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <iostream>



namespace nt2 { namespace ext
{
   BOOST_DISPATCH_IMPLEMENT( transform_, nt2::tag::opencl_<site>
                            , (A0)(A1)(site)
                            // , (boost::mpl::not_<
                            //     boost::mpl::and_<meta::is_container_terminal<A0>
                            //                     ,meta::is_container_terminal<A1>
                            //                     >
                            //                    >
                            //   )
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                            )
  {
    typedef void result_type;

    BOOST_FORCEINLINE result_type
    operator()(A0& a0, A1& a1) const
    {
      nt2::external_kernel<tag::transform_,tag::opencl_<site>>::call(a0, a1);
    }
  };
} }

#endif
#endif
