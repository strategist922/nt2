//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_DETAILS_CONV1D_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_CONV1D_HPP_INCLUDED

#include <nt2/include/functions/run.hpp>
#include <boost/simd/sdk/meta/iterate.hpp>

namespace nt2 { namespace details
{
  /*
    Perform a dynamic loop for computing a 1D convolution product
  */
  template<typename Out, typename In, typename K>
  BOOST_FORCEINLINE
  Out conv1D( int begin, int size, std::size_t limit
            , In const& in, K const& k
            )
  {
    typedef typename In::value_type   in_t;

    Out res = k.conv(nt2::run(in,begin,meta::as_<in_t>()),size,0);

    for(int j=1;j<limit;++j)
    {
      res = k.reduce( res
                    , k.conv( nt2::run(in,j+begin,meta::as_<in_t>())
                            , size, j
                            )
                    );
    }

    return k.normalize(res);
  }

  /*
    Static helper lambda
  */
  template<typename F, typename D, typename T, int Size>
  struct static_conv1D
  {
    BOOST_FORCEINLINE
    static_conv1D ( F const& f_, D const& data_, T& res_, int b, int s )
                  : f(f_), data(data_), res(res_), begin(b), size(s)
    {
    }

    template<int I> BOOST_FORCEINLINE void operator()() const
    {
      typedef boost::mpl::int_<I+1> I1;

      res = f.reduce( res , f.conv( nt2::run( data
                                            , begin + I1::value
                                            , meta::as_<T>()
                                            )
                                  , size, I1()
                                  )
                    );
    }

    F const& f;
    D const& data;
    T& res;
    int begin, size;

    private:
    static_conv1D& operator=(static_conv1D const&);
  };

  /*
    Perform a statically unrolled loop for computing a 1D convolution product
  */
  template<typename Out, typename T, std::size_t N, typename In, typename K>
  BOOST_FORCEINLINE
  Out conv1D( int begin, int size, boost::mpl::integral_c<T,N> const&
            , In const& in, K const& k
            )
  {
    typedef typename In::value_type   in_t;
    Out res = k.conv(nt2::run(in,begin,meta::as_<in_t>()),size,boost::mpl::int_<0>());

    static_conv1D<K,In,Out,N> stepper(k,in,res,begin,size);
    boost::simd::meta::iterate<N-1>(stepper);

    return k.normalize(res);
  }

} }

#endif
