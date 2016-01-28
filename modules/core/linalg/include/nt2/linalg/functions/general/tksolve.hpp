//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_GENERAL_TKSOLVE_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_GENERAL_TKSOLVE_HPP_INCLUDED

#include <nt2/linalg/functions/tksolve.hpp>
#include <nt2/core/container/table/table.hpp>
#include <nt2/include/functions/linsolve.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/eye.hpp>
#include <nt2/include/functions/sqr.hpp>
#include <nt2/include/functions/conj.hpp>
#include <nt2/include/functions/ctranspose.hpp>
#include <iostream>
namespace nt2{ namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( tksolve_, tag::cpu_
                            , (A0)(A1)
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                            )
  {
    typedef typename A0::value_type       value_type;
    typedef typename A0::index_type       index_type;
    typedef container::table<value_type, index_type> result_type;
    NT2_FUNCTOR_CALL(2)
      {
        return tksolve(a0, a1, 'N');
      }
  };

  BOOST_DISPATCH_IMPLEMENT  ( tksolve_, tag::cpu_
                            , (A0)(A1)(A2)
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                              (scalar_<integer_<A2> >)
                              )
  {
    typedef typename A0::value_type       value_type;
    typedef typename A0::index_type       index_type;
    typedef  container::table<value_type, index_type> result_type;
    NT2_FUNCTOR_CALL(3)
    {
      typedef container::table<value_type > tab_t;
      size_t n =  length(a0);
      tab_t a00 = a0;
      tab_t x = nt2::zeros(nt2::sqr(n), 1, nt2::meta::as_<value_type>());
      tab_t ii = eye(n, nt2::meta::as_<value_type>());
      std::cout << "a2 " << a2 << std::endl;
      if (a2 == 'N')
      {
        //    % Forward substitution.
        for (size_t i = 1; i <= n; ++i)
        {
          tab_t temp = a1(_(n*(i-1)+1, n*i));
          for (size_t j = 1; j <= i-1; ++j)
          {
            temp -= a00(j,i)*x(_(n*(j-1)+1, n*j));
          }
          x(_(n*(i-1)+1, n*i)) = nt2::linsolve((a00 + a00(i,i)*ii), temp);
        }
      }
      else if  (a2 == 'T')
      {
        //    Back substitution.
        for (size_t  i = n; i >= 1; --i)
        {
          tab_t temp = a1(_(n*(i-1)+1, n*i));
          for (size_t j = i+1; j <= n; ++j)
          {
            temp -= nt2::conj(a00(i,j))*x(_(n*(j-1)+1, n*j));
          }
          x(_(n*(i-1)+1, n*i)) = nt2::linsolve((ctrans(a00) + conj(a00(i,i))*ii), temp);
        }

      }
      return x;
    }
  };
} }

#endif
