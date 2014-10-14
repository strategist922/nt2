#ifndef NT2_CORE_FUNCTIONS_COMMON_DYNAMIC_STENCIL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_DYNAMIC_STENCIL_HPP_INCLUDED

#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/splat.hpp>
#include <boost/simd/include/functions/multiplies.hpp>

#include <nt2/include/functions/fma.hpp>
#include <nt2/include/functions/run.hpp>


#include <nt2/core/container/dsl/as_terminal.hpp>

#include <nt2/core/functions/common/dynamic_window.hpp>

namespace nt2 { namespace ext
{

  template< typename Expression> class dynamic_stencil
  {
    typedef typename Expression::extent_type extent_type;
    typedef typename Expression::value_type v_t;
    typedef memory::container<tag::table_,v_t,extent_type> sema_t;
    typedef typename container::as_terminal<sema_t, Expression>::type f_t;
    typedef typename boost::dispatch::meta
                     ::call<tag::numel_(extent_type const&)>::type size_type;

    dynamic_stencil& operator=(dynamic_stencil const&);

    public :

    template<typename T , typename Data>
    struct window
    {
      typedef dynamic_window<T,Data> type;
    };
    template<typename T , typename Data>
    struct window_simd
    {
      typedef dynamic_window<T,Data> type;
    };

    //Constructor
    BOOST_FORCEINLINE dynamic_stencil(Expression const& e) : stencil_(e) {}

    //C++11 auto? size getter.
    BOOST_FORCEINLINE size_type size() const
    {
      return numel( stencil_.extent() );
    }

    //Operation zones, define your stencil's operations.
    template<typename Out, typename Window>
    BOOST_FORCEINLINE
    Out operator()(  Window const& w, meta::as_< Out > const& ) const
    {
      const int size = this->size();
      Out res =  w(0,meta::as_<Out>() )*stencil_( size );

      for( int j = 1 ; j < size ; ++j)
      {
        res = nt2::fma( w( j , meta::as_<Out>() )
                      , boost::simd::splat<Out>(stencil_( size -j ) )
                      , res
                      );

      }
      return res;
    }

    //Operation zones, define your stencil's operations.
    template<typename Out, typename Window>
    BOOST_FORCEINLINE
    Out SIMD_operator(  Window const& w, meta::as_< Out > const& ) const
    {
      const int size = this->size();
      Out res =  w(0,meta::as_<Out>() )*stencil_( size );

      for( int j = 1 ; j < size ; ++j)
      {
        res = nt2::fma( w( j , meta::as_<Out>() )
                      , boost::simd::splat<Out>(stencil_( size -j ) )
                      , res
                      );

      }
      return res;
    }



    template< typename Out, typename Window >
    BOOST_FORCEINLINE
    Out operator()( Window const& w, meta::as_< Out > const& , const int begin
                    , const int size , const int limit ) const
    {
      Out res = w( begin , meta::as_<Out>() )
                *boost::simd::splat<Out>(stencil_( size ) );

      for( int j = 1 ; j < limit ; ++j)
      {
        res = nt2::fma( w( j + begin , meta::as_<Out>() )
                      , boost::simd::splat<Out>(stencil_( size -j ) )
                      , res
                      );
      }
      return res;
    }

      //Private
    private :
    f_t stencil_;
  };



}}
#endif
