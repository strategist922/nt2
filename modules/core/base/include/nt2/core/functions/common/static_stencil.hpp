#ifndef NT2_CORE_FUNCTIONS_COMMON_STATIC_STENCIL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_STATIC_STENCIL_HPP_INCLUDED

#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/splat.hpp>
#include <boost/simd/include/functions/multiplies.hpp>

#include <nt2/include/functions/fma.hpp>
#include <nt2/include/functions/run.hpp>

#include <nt2/core/container/dsl/as_terminal.hpp>

#include <nt2/core/functions/common/static_window.hpp>
#include <nt2/core/functions/simd/static_window_simd.hpp>

namespace nt2 { namespace ext
{

    //STATIC LOOP()
    template< typename Out , typename Window, typename Stencil, typename SizeA>
    struct static_stencil_loop
    {
      BOOST_FORCEINLINE
      static_stencil_loop( Out & out, Window const& w , Stencil const& s
                  , SizeA sA )
                 : result_(out) , window_(w), stencil_(s), size_(sA)
      {
      }

      template< int I > BOOST_FORCEINLINE
      void operator()() const
      {

        typedef boost::mpl::int_<I+1> I1;
        result_ = nt2::fma( window_( I1::value )
                          , boost::simd::splat<Out>(stencil_( size_ - I1::value))
                          , result_
                          );

      }

      Out & result_;
      Window const& window_;
      Stencil const& stencil_;
      SizeA size_;

    };
    //STATIC LOOP()
    template< typename Out , typename Window, typename Stencil, typename SizeA>
    struct static_stencil_loop_simd
    {
      BOOST_FORCEINLINE
      static_stencil_loop_simd( Out & out, Window const& w , Stencil const& s
                  , SizeA sA )
                 : result_(out) , window_(w), stencil_(s), size_(sA)
      {
      }

      template< int I > BOOST_FORCEINLINE
      void operator()() const
      {
        typedef boost::simd::meta::cardinal_of< Out > card_type;
        typedef boost::mpl::int_<I+1> I1;
        typedef boost::mpl::int_< I1::value / card_type::value > Id_register;
        typedef boost::mpl::int_< I1::value % card_type::value  > nb_of_slides;

        result_ = nt2::fma( boost::simd
                            ::slide<nb_of_slides::value
                                   >( window_(Id_register::value )
                                    , window_(Id_register::value+1 )
                                    )
                          , boost::simd::splat<Out>(stencil_( size_ - I1::value))
                          , result_
                          );

      }

      Out & result_;
      Window const& window_;
      Stencil const& stencil_;
      SizeA size_;

    };

  //STATIC STENCIL

  template< typename Expression> class static_stencil
  {

    typedef typename Expression::extent_type extent_type;
    typedef typename Expression::value_type v_t;
    typedef memory::container<tag::table_,v_t,extent_type> sema_t;
    typedef typename container::as_terminal<sema_t, Expression>::type f_t;
    typedef typename boost::dispatch::meta
                     ::call<tag::numel_(extent_type const&)>::type size_type;

    static_stencil& operator=(static_stencil const&);
    f_t stencil_;
    public :

    // CALLING THE GOOD WINDOW

    template<typename T, typename Data>
    struct window
    {
      typedef static_window<  T , Data , extent_type::static_numel > type;
    };

    template< typename T , typename Data >
    struct window_simd
    {
      typedef static_window_simd<  T , Data , extent_type::static_numel > type;
    };

    //Constructor
    BOOST_FORCEINLINE static_stencil(Expression const& e) : stencil_(e)
    {
    }

    //C++11 auto? size getter.
    BOOST_FORCEINLINE size_type size() const
    {
      return numel( stencil_.extent() );
    }

    //STATIC OPERATOR()
    template< typename Out, typename Window >
    BOOST_FORCEINLINE
    Out operator() ( Window const& w, meta::as_< Out > const& ) const
    {
      typedef boost::mpl::int_< extent_type::static_numel > sizeA;
      Out res = w( boost::mpl::int_<0>()  )
                *boost::simd::splat<Out>( stencil_(sizeA::value) );


      static_stencil_loop< Out , Window, f_t , int > stepper( res
                                                            , w , stencil_
                                                            , sizeA::value
                                                            );
      boost::simd::meta::iterate<sizeA::value - 1>(stepper);
      return res;
    }

    //STATIC SIMD OPERATOR
    template< typename Out, typename Window >
    BOOST_FORCEINLINE
    Out SIMD_operator ( Window const& w, meta::as_< Out > const& ) const
    {

      typedef boost::mpl::int_< extent_type::static_numel > sizeA;
      int indice_register;
      int indice_position;

      Out res = w( boost::mpl::int_<0>() )
                *boost::simd::splat<Out>( stencil_(sizeA::value) );

      static_stencil_loop_simd< Out , Window, f_t
                              , int > stepper( res
                                              , w , stencil_
                                              , sizeA::value
                                              );
      boost::simd::meta::iterate<sizeA::value - 1>(stepper);
      return res;
    }


    // BORDERS OPERATOR
    template< typename Out, typename Window >
    BOOST_FORCEINLINE
    Out operator()( Window const& w, meta::as_< Out > const& , const int begin
                    , const int size , const int limit ) const
    {
      Out res = w( begin  )*stencil_( size );

      for( int j = 1 ; j < limit ; ++j)
      {
        res = nt2::fma( w( j + begin ) , stencil_( size -j )
                      , res
                      );

      }
      return res;
    }
  };
}}
#endif
