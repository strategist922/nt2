//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_TRANSFORM_ALONG_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_TRANSFORM_ALONG_HPP_INCLUDED

#include <nt2/include/functor.hpp>

namespace nt2
{
  namespace tag
  {
   /*!
     @brief transform_along generic tag

     Represents the transform_along function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct transform_along_ : boost::dispatch::tag::formal_
    {
      typedef boost::dispatch::tag::formal_ parent;
    };
  }

  /*!

   **/
  NT2_FUNCTION_IMPLEMENTATION(tag::transform_along_, transform_along, 5)

  /// @overload
  NT2_FUNCTION_IMPLEMENTATION(tag::transform_along_, transform_along, 4)

  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0&)(A1 const&)
                                              (A2 const&)(A3 const&)
                                            , 4
                                            )
  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0 const&)(A1&)
                                              (A2 const&)(A3 const&)
                                            , 4
                                            )
  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0&)(A1&)
                                              (A2 const&)(A3 const&)
                                            , 4
                                            )

  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0&)(A1 const&)
                                              (A2 const&)(A3 const&)
                                              (A4 const&)
                                            , 5
                                            )
  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0 const&)(A1&)
                                              (A2 const&)(A3 const&)
                                              (A4 const&)
                                            , 5
                                            )
  /// @overload
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL( tag::transform_along_
                                            , transform_along
                                            , (A0&)(A1&)
                                              (A2 const&)(A3 const&)
                                              (A4 const&)
                                            , 5
                                            )
}

#endif
