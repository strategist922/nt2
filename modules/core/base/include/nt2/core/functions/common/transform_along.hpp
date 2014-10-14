//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_COMMON_TRANSFORM_ALONG_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_TRANSFORM_ALONG_HPP_INCLUDED

#include <nt2/include/functions/run.hpp>

#include <nt2/core/functions/common/static_stencil.hpp>
#include <nt2/core/functions/common/dynamic_stencil.hpp>

namespace nt2 { namespace ext
{

  //============================================================================
  // Global version
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_along_, tag::cpu_
                            , (Out)(In)(K)(Shp)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<K>)
                              (unspecified_<Shp>)
                            )
  {
    typedef void result_type;
    result_type operator()( Out& out, In& in
                          , K const& kernel, Shp const& s ) const
    {
      transform_along ( out, in, kernel
                      , std::make_pair( 0, out.size() )
                      , s
                      );
    }
  };
  //============================================================================
  // Ranged version
  //============================================================================
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_along_, tag::cpu_
                            , (Out)(In)(K)(Rng)(Shp)
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
    BOOST_FORCEINLINE
    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::valid_> const&
                          ) const
    {

      typedef typename In::value_type in_t;
      int start = r.first;
      const int end  = r.first + r.second - 1  ;

      if(end >= 0)
      {

        typename K::template window<in_t ,In>::type Window_( in  , 0);
        typedef typename Out::value_type  out_t;
        main_loop( out, in , Window_ , kernel , start , end );
        Window_.load();
        nt2::run(out,end, kernel( Window_, meta::as_<out_t>() ));

      }
    }

    //MAIN LOOP
    template< typename W >
    BOOST_FORCEINLINE
    result_type main_loop(Out & out , In & in , W & window_ , K const & kernel
                         , int & start ,  int end
                         )const
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
    BOOST_FORCEINLINE
    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::same_> const&
                          ) const
    {
      typedef typename Out::value_type  out_t;
      typedef typename In::value_type in_t;
      int start = r.first;

      //PROLOGUE => 1 WINDOW LOADING
      const int n = kernel.size();
      int kerD, kerG;
      kerG = (n-(n%2))/2 - (1-n%2);
      kerD = n/2 ;

      if( n < out.size() )
      {

        typename K::template window<in_t , In>::type Window_( in  , start );
        int end = kerG ;
        for( ; start < end ; ++start )
        {

           nt2::run(out,start,kernel( Window_ , meta::as_<out_t>() , 0
                                    , start + kerD + 1 , start + kerD + 1
                                    )
                    );

        }

      //MAIN LOOP
        end = r.second - kerD;
        main_loop( out, in , Window_ , kernel , start , end );
        Window_.load();

      //EPILOGUE => 1 window Loading
        end = r.first + r.second;
        for(int counter = kerD + kerG, t = 0 ;  start < end
            ; ++t , ++start , --counter
            )
        {

          nt2::run(out,start,kernel( Window_ , meta::as_<out_t>()
                                   , t , n , counter
                                   )
                  );

        }
      }
      else
      {

        int end =r.first + r.second;
        typename K::template window<in_t , In>::type Window_( in  , kerG );

        for(int pp = kerG; start < end ; ++start, ++pp )
        {

           Window_.load();
           nt2::run(out,start,kernel( Window_ , meta::as_<out_t>() ));
           Window_.update();

        }
      }
    }

  //============================================================================
  //Version Full
  //============================================================================
    BOOST_FORCEINLINE
    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          , policy<ext::full_> const&
                          ) const
    {
      typedef typename Out::value_type  out_t;
      typedef typename In::value_type in_t;
      int start = r.first;
      typename K::template window<in_t , In>::type Window_( in  , start );

      //PROLOGUE => 1 WINDOW LOADING
      int n = kernel.size();
      int end = start + n - 1 ;
      typedef typename Out::value_type  out_t;

       for( ; start < end ; ++start )
       {

         nt2::run(out,start,kernel( Window_ , meta::as_<out_t>(), 0
                                  , start + 1 , start + 1
                                  )
                 );
       }

       //MAIN LOOP
       end = r.first + r.second - n + 1;
       main_loop( out, in , Window_ , kernel , start , end );
       Window_.load();
       int temp = out.size();

       //EPILOGUE => 1 Window Loading
       end =r.first + r.second;
       for( int counter = 0 ; start < end ; ++start , ++counter )
       {
         nt2::run(out,start,kernel( Window_ , meta::as_<out_t>(), counter
                                  , n ,  min(n , temp - start)
                                  )
                 );
       }

      }

  };
} }



#endif
