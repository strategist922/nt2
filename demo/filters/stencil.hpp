//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_STENCIL_HPP_INCLUDED
#define FILTERS_STENCIL_HPP_INCLUDED

namespace nt2
{
  namespace tag
  {
    struct regular_stencil_                       {};
    struct reductible_stencil_ : regular_stencil_ {};
  }

  // Compact stencils load every point in their structuring element
  struct compact_
  {
    template<std::size_t Row, std::size_t Col>
    struct in_stencil : boost::mpl::true_
    {};
  };
}

#endif
