/*******************************************************************************
 *         Copyright 2003-2010 LASMEA UMR 6602 CNRS/U.B.P
 *         Copyright 2009-2010 LRI    UMR 8623 CNRS/Univ Paris Sud XI
 *
 *          Distributed under the Boost Software License, Version 1.0.
 *                 See accompanying file LICENSE.txt or copy at
 *                     http://www.boost.org/LICENSE_1_0.txt
 ******************************************************************************/
#ifndef NT2_TOOLBOX_TRIGONOMETRIC_FUNCTION_COMMON_SIMD_IMPL_TRIGO_D_TRIG_REDUCTION_HPP_INCLUDED
#define NT2_TOOLBOX_TRIGONOMETRIC_FUNCTION_COMMON_SIMD_IMPL_TRIGO_D_TRIG_REDUCTION_HPP_INCLUDED

#include <nt2/sdk/meta/logical.hpp>
#include <nt2/include/functions/rem_pio2_medium.hpp>
#include <nt2/include/functions/rem_pio2_cephes.hpp>
#include <nt2/include/functions/rem_pio2_straight.hpp>
#include <nt2/include/functions/rem_pio2.hpp>
//#include <nt2/toolbox/trigonometric/function/simd/common/impl/trigo/f_pio2_reducing.hpp>
#include <nt2/include/functions/is_odd.hpp>
#include <nt2/include/functions/is_even.hpp>
#include <nt2/include/functions/is_invalid.hpp>
#include <nt2/include/functions/inrad.hpp>
#include <nt2/include/functions/rec.hpp>
#include <nt2/include/constants/digits.hpp>
#include <nt2/include/constants/real.hpp>

namespace nt2
{
  namespace details
  {
    namespace internal
    {
      // trigonometric reduction strategies in the [-pi/4, pi/4] range.
      // these reductions are used in the accurate and fast
      // trigonometric functions with different policies
      template < class A0, class precision_tag, class mode>
      struct trig_reduction < A0, radian_tag, precision_tag, tag::simd_type, mode, double>
      {
        typedef typename meta::as_integer<A0, signed>::type int_type;
        typedef typename meta::scalar_of<int_type>::type   sint_type;
        typedef typename meta::logical<A0>::type               logic;

        static inline logic isalreadyreduced(const A0&a0){ return le(a0, Pio_4<A0>()); }
        static inline logic ismedium (const A0&a0){return le(a0,double_constant<A0,0x412921fb54442d18ll>()); }
        static inline logic issmall  (const A0&a0){return le(a0,double_constant<A0,0x404f6a7a2955385ell>()); }
        static inline logic islessthanpi_2  (const A0&a0)  { return le(a0,Pio_2<A0>()); }
	static inline logic cot_invalid(const A0& x) { return False<A0>()/*is_invalid(x)*/; }
	static inline logic tan_invalid(const A0& x) { return False<A0>()/*is_invalid(x)*/; }
        static inline int_type reduce(const A0& x, A0& xr, A0& xc){ return inner_reduce(x, xr, xc, mode()); }
      private:
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const big&)
        {
          // x is always positive here
          if (all(isalreadyreduced(x))) // all of x are in [0, pi/4], no reduction
            {
	      xr = x;
	      xc = Zero<A0>();
	      return Zero<int_type>(); 
            }
	  else if (all(islessthanpi_2(x))) // all of x are in [0, pi/2],  straight algorithm is sufficient for 1 ulp
	    return rem_pio2_straight(x, xr, xc);
          else if (all(issmall(x))) // all of x are in [0, 20*pi],  cephes algorithm is sufficient for 1 ulp
	    return rem_pio2_cephes(x, xr, xc);
          else if (all(ismedium(x))) // all of x are is in [0, 2^18*pi],  fdlibm medium way
	    return rem_pio2_medium(x, xr, xc);
	  else  // all of x are in [0, inf],  standard big way
	    return rem_pio2(x, xr, xc);
        }
	
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const medium&)
        {
          // x is always positive here
          if (all(isalreadyreduced(x))) // all of x are in [0, pi/4], no reduction
            {
	      xr = x;
	      xc = Zero<A0>();
	      return Zero<int_type>(); 
            }
	  else if (all(islessthanpi_2(x))) // all of x are in [0, pi/2],  straight algorithm is sufficient for 1 ulp
	    return rem_pio2_straight(x, xr, xc);
          else if (all(issmall(x))) // all of x are in [0, 20*pi],  cephes algorithm is sufficient for 1 ulp
	    return rem_pio2_cephes(x, xr, xc);
          else  // correct only if all of x are is in [0, 2^18*pi],  fdlibm medium way
	    return rem_pio2_medium(x, xr, xc);
        }
	
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const small&)
        {
          // x is always positive here
          if (all(isalreadyreduced(x))) // all of x are in [0, pi/4], no reduction
            {
	      //	      std::cout << "1111111111111" << std::endl; 
	      xr = x;
	      xc = Zero<A0>();
	      return Zero<int_type>(); 
            }
	  else if (all(islessthanpi_2(x))) // all of x are in [0, pi/2],  straight algorithm is sufficient for 1 ulp
	    {
	      //	      std::cout << "2222222222222" << std::endl; 
	      return rem_pio2_straight(x, xr, xc);
	    }
          else  // correct only if all of x are in [0, 20*pi],  cephes algorithm is sufficient for 1 ulp
	    {
	      //	      std::cout << "333333333333" << std::endl;
	      //	      A0 x1 = b_or(x, is_invalid(x));
	      return rem_pio2_cephes(x, xr, xc);
	    }
	}
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const direct_cephes&)
        {
	  return rem_pio2_cephes(x, xr, xc);
        }
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const direct_medium&)
        {
	  return rem_pio2_medium(x, xr, xc);
        }
        static inline int_type inner_reduce(const A0& x, A0& xr, A0& xc, const direct_big&)
        {
	  return rem_pio2_big(x, xr, xc);
        }
      };
      


      
//      template < class A0 > struct trig_reduction < A0, radian_tag, fast_tag, tag::simd_type, double>
//       {
// 	typedef typename meta::as_integer<A0, signed>::type int_type;
// 	typedef typename meta::scalar_of<int_type>::type   sint_type;
// 	typedef typename meta::logical<A0>::type               logic;

// 	//static const sint_type Med  = 0x412921fb54442d18ll; //2^18*pi   8.235496645826427e+05
// 	static inline logic replacement_needed(const A0& a0) {return False<A0>();ignore_unused(a0); }//return a0 > double_constant<A0, Med>();}//
// 	static inline logic replacement_available(){return True<A0>(); }
// 	static inline logic isalreadyreduced(const A0&a0){return lt(a0,Pio_4<A0>()); }
// 	static inline logic issmall  (const A0&a0){return le(a0,double_constant<A0,0x404f6a7a2955385ell>()); }
// 	static inline A0 cos_replacement(const A0& a0){ return std::cos(a0); }
// 	static inline A0 sin_replacement(const A0& a0){ return std::sin(a0); }
// 	static inline A0 tan_replacement(const A0& a0){ return std::tan(a0); }
// 	static inline A0 cot_replacement(const A0& a0){ return rec(std::tan(a0)); }
//         static inline logic cot_invalid(const A0& x) { return False<A0>(); /*is_eqz(x)|is_invalid(x);*/ }
//         static inline logic tan_invalid(const A0& x) { return False<A0>(); /*is_invalid(x);*/ }
// 	static inline void sincos_replacement(const A0& a0, A0&s, A0&c){ }//::sincosf(a0, &s, &c); }
// 	static inline int_type reduce(const A0& x, A0& xr, A0& xc)
// 	{
// 	  // x is always positive here
// 	  // In all case we apply the cephes reduction scheme
// 	  // This impies good reduction in [0 64*pi]
// 	  // and degradation near pi/2 multiples on greater values
// 	  return pio2_reducing<A0, tag::simd_type>::cephes_reduction(x, xr, xc);
//  	}
//      };

      template < class A0>
      struct trig_reduction < A0, degree_tag, trig_tag, tag::simd_type, big, double >
      {
	typedef typename meta::as_integer<A0, signed>::type  int_type;
	typedef typename meta::scalar_of<int_type>::type    sint_type;
	typedef typename meta::logical<A0>::type                logic;

	static inline logic replacement_needed(const A0& a0) {return gt(a0,double_constant<A0,0x416fffffe0000000ll>());}
	static inline logic replacement_available(){return True<A0>(); }
	static inline A0 cos_replacement(const A0& a0){ return std::cos(inrad(a0)); }
	static inline A0 sin_replacement(const A0& a0){ return std::sin(inrad(a0)); }
	static inline A0 tan_replacement(const A0& a0){ return std::tan(inrad(a0)); }
	static inline A0 cot_replacement(const A0& a0){ return rec(std::tan(inrad(a0))); }
	static inline void sincos_replacement(const A0& a0, A0&s, A0&c){ }//::sincos(inrad(a0), &s, &c); }
        static inline logic cot_invalid(const A0& x) { return /*is_invalid(x)|*/(is_nez(x)&is_even(x/Ninety<A0>())); }
        static inline logic tan_invalid(const A0& x) { return /*is_invalid(x)|*/is_odd(x/Ninety<A0>()); }
	static inline int_type reduce(A0 x, A0& xr, A0& xc)
	{
	  A0 xi = round2even(x*double_constant<A0,0x3f86c16c16c16c17ll>());//1.111111111111111e-02
	  A0 x2 = x - xi * Ninety<A0>();//90.0f
	  xr =  x2*double_constant<A0,0x3f91df46a2529d39ll>();//0.0174532925199432957692
	  xc = Zero<A0>();
	  return toint(xi);
	}
      };
      
      template < class A0>
      struct trig_reduction < A0, pi_tag, trig_tag, tag::simd_type, big, double>
      {
	typedef typename meta::as_integer<A0, signed>::type int_type;
	typedef typename meta::scalar_of<int_type>::type    sint_type;
	typedef typename meta::logical<A0>::type                logic;

	// floats x greater than 2^52 are integers so
	// sin(pi*x) and tan(pi*x) are thus 0
	// cos(pi*x) is thus (-1)^x
	// cotpi*x) is thus Nan
	static inline logic replacement_needed(const A0& ) {return False<A0>(); }//return a0 > double_constant<A0,bm::integral_c<uint64_t, Big> >();}
	static inline logic replacement_available(){return True<A0>(); }
	static inline A0 cos_replacement(const A0& a0){ return One<A0>()-(is_odd(a0)<<1); }
	static inline A0 sin_replacement(const A0& ){ return Zero<A0>(); }
	static inline A0 tan_replacement(const A0& ){ return Zero<A0>(); }
	static inline A0 cot_replacement(const A0& ){ return Nan<A0>();  }
	static inline void sincos_replacement(const A0& a0, A0&s, A0&c){ c = cos_replacement(a0); s = sin_replacement(a0); }
        static inline logic cot_invalid(const A0& x) { return /*is_invalid(x)|*/(is_nez(x)&is_flint(x)); }
        static inline logic tan_invalid(const A0& x) { return /*is_invalid(x)|*/is_flint(x-Half<A0>()); }
	static inline int_type reduce(const A0& x,  A0& xr, A0&xc)
	{
	  A0 xi = round2even(x*Two<A0>());
	  A0 x2 = x - xi * Half<A0>();
	  xr = x2*Pi<A0>();
	  xc = Zero<A0>();
	  return toint(xi);
	}
      };
    }
  }
}


#endif

// /////////////////////////////////////////////////////////////////////////////
// End of d_trig_reduction.hpp
// /////////////////////////////////////////////////////////////////////////////
