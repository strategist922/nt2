//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_GENERAL_SQRTM_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_GENERAL_SQRTM_HPP_INCLUDED

#include <nt2/linalg/functions/sqrtm.hpp>
#include <nt2/core/container/table/table.hpp>
#include <nt2/include/functions/tofloat.hpp>
#include <nt2/include/functions/sqrt.hpp>
#include <nt2/include/functions/diag_of.hpp>
#include <nt2/include/functions/from_diag.hpp>
#include <nt2/include/functions/isdiagonal.hpp>
#include <nt2/include/functions/schur.hpp>
#include <nt2/include/functions/mtimes.hpp>
#include <nt2/include/functions/globalsum.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/eye.hpp>
#include <nt2/include/functions/length.hpp>
#include <nt2/include/functions/transpose.hpp>
#include <nt2/include/functions/conj.hpp>
#include <nt2/include/functions/real.hpp>
#include <nt2/include/functions/colvect.hpp>
#include <nt2/include/functions/cast.hpp>
#include <complex>

namespace nt2{ namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( sqrtm_, tag::cpu_
                            , (A0)
                            , ((ast_<A0, nt2::container::domain>))
                            )
  {
    typedef typename A0::value_type            value_type;
    typedef typename A0::index_type            index_type;
    typedef table<value_type, index_type>     result_type;
    NT2_FUNCTOR_CALL(1)
    {
      return doit(a0, typename meta::is_complex<value_type>::type());
    }
  private :

    result_type doit(const A0 & a0, boost::mpl::false_ const &) const
    {
      typedef typename std::complex<value_type>   cmplx_type;
      typedef nt2::table<cmplx_type, nt2::_2D>       ctab_t;
      size_t n = length(a0);
      ctab_t q, t, r;
      tie(q, t) = schur(a0, nt2::cmplx_);
      compute(n, t, r);
      ctab_t x = nt2::mtimes( nt2::mtimes(q, r), nt2::trans(nt2::conj(q)));
      return nt2::real(x);
      //bool nzeig = any(diag_of(t)(_))(1);

      // if nzeig
      //     warning(message('sqrtm:SingularMatrix'))
      // end
    }

    result_type doit(const A0 & a0, boost::mpl::true_ const &) const
    {
      typedef nt2::table<value_type, nt2::_2D>        tab_t;
      size_t n = length(a0);
      tab_t q, t, r;
      tie(q, t) = schur(a0, nt2::cmplx_);
      compute(n, t, r);
      return nt2::mtimes( nt2::mtimes(q, r), nt2::trans(nt2::conj(q)));

      //bool nzeig = any(diag_of(t)(_))(1);

      // if nzeig
      //     warning(message('sqrtm:SingularMatrix'))
      // end
    }
    template < class S > void compute(const size_t & n, const S& t, S& r) const
    {
      typedef typename S::value_type v_type;
      if (nt2::isdiagonal(t))
      {
        r = nt2::from_diag(nt2::sqrt(nt2::diag_of(t)));
      }
      else
      {
        // Compute upper triangular square root R of T, a column at a time.
        r = nt2::zeros(n, n, meta::as_<v_type>());
        for (size_t j=1; j <= n; ++j)
        {
          r(j,j) = nt2::sqrt(t(j,j));
          for (size_t i=j-1; i >= 1; --i)
          {
            //itab_ k = _(i+1, j-1);
            v_type s = nt2::globalsum(nt2::multiplies(colvect(r(i,_(i+1, j-1))),
                                                      colvect(r(_(i+1, j-1),j))));
            //                  value_type s = nt2::mtimes(r(i,_(i+1, j-1)), r(_(i+1, j-1),j));
            r(i,j) = (t(i,j) - s)/(r(i,i) + r(j,j));
          }
        }
      }
    }

  };

  BOOST_DISPATCH_IMPLEMENT  ( sqrtm_, tag::cpu_
                            , (A0)
                            , (scalar_<fundamental_<A0> >)
                            )
  {
    typedef A0 result_type;
    NT2_FUNCTOR_CALL(1)
    {
      return nt2::sqrt(a0);
    }
  };
} }

#endif

