//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_WINDOW_HELPERS_HPP_INCLUDED
#define FILTERS_WINDOW_HELPERS_HPP_INCLUDED

#include <cstddef>
#include <boost/dispatch/attributes.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/include/functions/simd/slide.hpp>
#include <boost/simd/include/functions/simd/aligned_load.hpp>

namespace nt2 { namespace details
{
  // Data sliding for register reduction optimization
  template< typename Operator
          , typename Type
          , std::size_t Index = 0
          , std::size_t Size  = Operator::width/2
          , std::size_t Card  = boost::simd::meta::cardinal_of<Type>::value
          >
  struct slide_
  {
    template<typename Data> BOOST_FORCEINLINE void static call(Data& d)
    {
      using boost::simd::slide;

      static const std::ptrdiff_t N  = Size-Index;

      // Compute how much we slide accounting width greater than cardinal
      static const std::size_t    SP = (N % Card);
      static const std::size_t    SN = -SP;

      // Compute which registers slide accounting width greater than cardinal
      static const std::size_t    X  = (N-1)/Card;
      static const std::size_t    X1 = X+1;
      static const std::size_t    X2 = X+2;

      // Symmetrical sliding
      d[Index]                   = slide<SN>(d[X1],d[X ]);
      d[Operator::width-Index-1] = slide<SP>(d[X1],d[X2]);

      slide_<Operator,Type,Index+1,Size,Card>::call(d);
    }
  };

  // Scalars don't slide
  template<typename Operator, typename Type, std::size_t I,std::size_t S>
  struct slide_<Operator,Type,I,S,1u>
  {
    template<typename Data> BOOST_FORCEINLINE void static call(Data& ) {}
  };

  // End recursion on Index/Size
  template<typename Operator, typename Type, std::size_t S, std::size_t C>
  struct slide_<Operator,Type,S,S,C>
  {
    template<typename Data> BOOST_FORCEINLINE void static call(Data& ) {}
  };

  template< typename Operator, typename Type, std::size_t Size>
  struct slide_<Operator,Type,Size,Size,1u>
  {
    template<typename Data> BOOST_FORCEINLINE void static call(Data& ) {}
  };

  // Circular rotation of data
  template<std::size_t Size,std::size_t Iter = 0>
  struct circ_shift_
  {
    template<typename Data> BOOST_FORCEINLINE static void call( Data& d )
    {
      d[Iter] = d[Iter+1];
      circ_shift_<Size,Iter+1>::call(d);
    }
  };

  template<std::size_t Size> struct circ_shift_<Size,Size>
  {
    template<typename Data> BOOST_FORCEINLINE static void call( Data& ) {}
  };

  // Fill up a window statically with respect to the
  // structuring element shape and data type
  template< typename Operator, typename Type
          , std::size_t J = 0, std::size_t I = 0
          , bool nextLine = (I == Operator::width)
          , bool stop     = nextLine && (J==Operator::height-1)
          , bool in_stencil = Operator::template in_stencil<J,I>::type::value
          >
  struct extract_window
  {
    static const std::ptrdiff_t j_offset = J - Operator::height/2;
    static const std::ptrdiff_t i_offset = I - Operator::width/2;
    static const std::size_t    d_offset = I+J*Operator::width;

    template<typename Data, typename Source> BOOST_FORCEINLINE
    void static call(std::size_t j, std::size_t i, Data const&d, Source const& src)
    {
      using boost::simd::aligned_load;

      d[d_offset] = aligned_load<Type,i_offset>(&src[j+j_offset][i+i_offset]);
      extract_window<Operator,Type,J,I+1>::call(j,i,d,src);
    }
  };

  template< typename Operator, typename Type, std::size_t J, std::size_t I>
  struct extract_window<Operator,Type,J,I,false,false,false>
  {
    template<typename Data, typename Source> BOOST_FORCEINLINE
    void static call(std::size_t j, std::size_t i, Data d, Source const& src)
    {
      extract_window<Operator,Type,J,I+1>::call(j,i,d,src);
    }
  };

  template< typename Operator, typename Type
          , std::size_t J, std::size_t I, bool in_stencil
          >
  struct extract_window<Operator,Type,J,I,true,false,in_stencil>
  {
    template<typename Data, typename Source> BOOST_FORCEINLINE
    void static call(std::size_t j, std::size_t i, Data d, Source const& src)
    {
      extract_window<Operator,Type,J+1,0>::call(j,i,d,src);
    }
  };

  template< typename Operator, typename Type
          , std::size_t J, std::size_t I, bool in_stencil
          >
  struct extract_window<Operator,Type,J,I,true,true, in_stencil>
  {
    template<typename Data, typename Source> BOOST_FORCEINLINE
    void static call(std::size_t,std::size_t,Data, Source const&) {}
  };

  // Unroll operation over element of a structuring element
  template< typename Operator
          , std::size_t Iter = Operator::height*Operator::width
          >
  struct fold_
  {
    template<typename Data>
    BOOST_FORCEINLINE typename Data::value_type static call(Data const& d)
    {
      return Operator::call(d[Iter-1], fold_<Operator,Iter-1>::call(d));
    }
  };

  template<typename Operator> struct fold_<Operator,2>
  {
    template<typename Data>
    BOOST_FORCEINLINE typename Data::value_type static call(Data const& d)
    {
      return Operator::call(d[1], d[0]);
    }
  };

  template<typename Operator> struct fold_<Operator,1>
  {
    template<typename Data>
    BOOST_FORCEINLINE typename Data::value_type static call(Data const& d)
    {
      return d[0];
    }
  };
} }

#endif
