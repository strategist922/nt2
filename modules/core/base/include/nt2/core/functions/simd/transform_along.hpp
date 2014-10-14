//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_ALONG_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_ALONG_HPP_INCLUDED

#include <boost/simd/sdk/meta/cardinal_of.hpp>

#include <boost/simd/sdk/simd/native.hpp>
#include <boost/simd/sdk/simd/meta/vector_of.hpp>
#include <boost/simd/sdk/simd/meta/is_vectorizable.hpp>

#include <nt2/core/functions/common/static_stencil.hpp>
#include <nt2/core/functions/common/dynamic_stencil.hpp>

#include <nt2/include/functions/run.hpp>

namespace nt2 { namespace ext
{
  //============================================================================
  // Ranged version
  //============================================================================
    NT2_FUNCTOR_IMPLEMENTATION_IF( nt2::tag::transform_along_
                            , boost::simd::tag::simd_
                            , (Out)(In)(K)(Rng)(Shp)
                            ,  (mpl::and_< boost::simd::meta::
                                          is_vectorizable<typename Out::value_type
                                          , BOOST_SIMD_DEFAULT_EXTENSION>
                                          , boost::simd::meta::
                                          is_vectorizable<typename In::value_type
                                          , BOOST_SIMD_DEFAULT_EXTENSION> >)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<K>)
                              (unspecified_<Rng>)
                              (unspecified_<Shp>)
                              )
    {
  //============================================================================
  //Version valid
  //============================================================================
    typedef void result_type;

    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::valid_> const&
                          ) const
    {
      int start = r.first;
      const int end  = r.first + r.second   ;
      typedef typename Out::value_type  out_t;
      typedef boost::simd::native<out_t, BOOST_SIMD_DEFAULT_EXTENSION> target_type;
      typedef typename In::value_type in_t;
      typedef typename boost::simd::meta
                          ::vector_of < in_t
                                      , boost::simd::meta
                                             ::cardinal_of<target_type>::value
                                      >::type in_tt;

      std::size_t card = boost::simd::meta::cardinal_of<in_tt>::value;

      //We need to check if my data is large enough for SIMD
      if(card*kernel.size() <= r.second)
      {

        typename K::template window_simd<in_tt,In>::type siw( in  , start );
        main_loop_simd( out, in , siw , kernel , start , end );

      }

      typename K::template window<in_t,In>::type scw( in  , start );
      main_loop( out, in , scw, kernel , start , end );

    }

    //MAIN LOOP SIMD FUNCTION
    template< typename W >
    BOOST_FORCEINLINE
    result_type main_loop_simd(Out & out , In & in , W & window_
                              , K const & kernel, int & start ,  int end
                              )const
    {
      typedef typename Out::value_type  out_t;
      typedef boost::simd::native<out_t, BOOST_SIMD_DEFAULT_EXTENSION> target_type;
      std::size_t card = boost::simd::meta::cardinal_of<target_type>::value;
      std::size_t aligned_sz  = std::max((end-start),0) & ~(card-1);
      std::size_t mm = start + aligned_sz - (kernel.size()/card)*card;

      for(; start < mm ; start += card )
      {

        window_.load();
        nt2::run(out,start,kernel.SIMD_operator( window_
                                               , meta::as_<target_type>()
                                               )
                );
        window_.update();

      }
    }

    //MAIN LOOP SCALAR FUNCTION
    template< typename W >
    BOOST_FORCEINLINE
    result_type main_loop( Out & out , In & in , W & window_ , K const & kernel
                         , int& start ,  int end
                         ) const
    {
      typedef typename Out::value_type  out_t;
      for( ; start < end  ; ++start )
      {

        window_.load();
        nt2::run(out , start , kernel( window_, meta::as_<out_t>() ));
        window_.update();

      }
    }

  //============================================================================
  //Version same
  //============================================================================

    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::same_> const&
                          ) const
    {
      typedef typename Out::value_type  out_t;
      typedef typename In::value_type in_t;

      int n = kernel.size();
      int kerD, kerG;
      kerG = (n-(n%2))/2 - (1-n%2);
      kerD = n/2 ;
      int start = r.first ;

      if( n < out.size() )
      {

        //PROLOGUE => 1 WINDOW LOADING
        typename K::template window<in_t,In>::type Window_( in  , start );
        int end = kerG ;

        for( ; start < end ; ++start )
        {

          nt2::run(out,start,kernel( Window_ , meta::as_<out_t>() , 0
                                    , start + kerD + 1 , start + kerD + 1
                                    )
                  );

       }

        //MAIN LOOP CALL
        end = r.second - kerD;
        typedef boost::simd::native< out_t
                                   ,  BOOST_SIMD_DEFAULT_EXTENSION
                                   > target_type;
        typedef typename In::value_type in_t;
        typedef typename boost::simd::meta
                            ::vector_of < in_t
                                        , boost::simd::meta
                                               ::cardinal_of<target_type>::value
                                        >::type in_tt;
        std::size_t card = boost::simd::meta::cardinal_of<in_tt>::value;

        //We need to check if my data is large enough for SIMD
        if(card*kernel.size() <= end - start)
        {

          typename K::template window<in_tt,In>::type siw( in  , start - kerG );
          main_loop_simd( out, in , siw , kernel , start , end );

        }

        typename K::template window<in_t,In>::type scw( in  , start - kerG );
        main_loop( out, in , scw, kernel , start , end );
        scw.load();

        //EPILOGUE  => 1 WINDOW LOADING
        start = end;
        end = r.first + r.second;

        for(int counter = kerD + kerG, t = 0 ;  start < end
            ; ++t , ++start , --counter )
        {

          nt2::run(out,start,kernel( scw , meta::as_<out_t>()
                                   , t , n , counter
                                   )
                  );

        }
      }
      else
      {

        int end =r.first + r.second;
        int pp =  kerG;
        typename K::template window<in_t,In>::type Window_( in  ,  pp  );

        for(; start < end ; ++start, ++pp )
        {

          Window_.load();
          nt2::run(out,start,kernel( Window_ , meta::as_<out_t>() ) );
          Window_.update();

        }
      }
    }

  //============================================================================
  //Version Full
  //============================================================================

    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::full_> const&
                          ) const
    {

      typedef typename Out::value_type  out_t;
      typedef typename In::value_type in_t;
      int start = r.first;
      typename K::template window< in_t , In >::type Window_( in  , start );

      int n = kernel.size();
      int end = start + n - 1 ;
      typedef typename Out::value_type  out_t;

      //PROLOGUE => 1 WINDOW LOADING
      for( ; start < end ; ++start )
      {
       nt2::run(out,start,kernel( Window_ , meta::as_<out_t>(), 0
                                , start + 1 , start + 1
                                )
               );

      }

      //MAIN LOOP CALL
       end = r.first + r.second - n + 1;
      typedef boost::simd::native<out_t, BOOST_SIMD_DEFAULT_EXTENSION> target_type;
      typedef typename In::value_type in_t;
      typedef typename boost::simd::meta
                          ::vector_of < in_t
                                      , boost::simd::meta
                                             ::cardinal_of<target_type>::value
                                      >::type in_tt;
      std::size_t card = boost::simd::meta::cardinal_of<in_tt>::value;

      //We need to check if my data is large enough for SIMD
      if(card*kernel.size() <= end - start)
      {

        typename K::template window<in_tt,In>::type siw( in  ,  start - n + 1);
        main_loop_simd( out, in , siw , kernel , start , end );

      }

      typename K::template window<in_t,In>::type scw( in  , start - n +1 );
      main_loop( out, in , scw, kernel , start , end );
      scw.load();
       int temp = out.size();


      //EPILOGUE  => 1 WINDOW LOADING
      end =r.first + r.second;
      for( int counter = 0 ; start < end ; ++start , ++counter )
      {

         nt2::run(out,start,kernel( scw , meta::as_<out_t>(), counter
                                  , n ,min( n , temp - start )
                                  )
                 );

      }
     }
  };
} }
#endif
