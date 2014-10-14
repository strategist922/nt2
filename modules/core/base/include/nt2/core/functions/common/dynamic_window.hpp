#ifndef NT2_CORE_FUNCTIONS_COMMON_DYNAMIC_WINDOW_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_DYNAMIC_WINDOW_HPP_INCLUDED

#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/sdk/simd/meta/vector_of.hpp>

#include <nt2/include/functions/run.hpp>

namespace nt2 { namespace ext
{
  //////////////////////////////////////////////////////////////////////////////
  //DYNAMIC WINDOW
  //////////////////////////////////////////////////////////////////////////////

  template< typename T, typename In > class dynamic_window
  {
    dynamic_window& operator=(dynamic_window const&);

    public :
    size_t  in_position_;
    In const& data_;

    typedef typename In::value_type value_type;
    typedef boost::simd::meta::cardinal_of<T> card_type;


    //CONSTRUCTOR
    BOOST_FORCEINLINE
    dynamic_window( In const& d, size_t begin )
                  : in_position_( begin )
                  , data_(d)
    {
    }

    BOOST_FORCEINLINE void update()
    {
      in_position_ += card_type::value;
    }


    BOOST_FORCEINLINE void load()
    {
    }

    template<typename Out>
    Out operator()(size_t p, meta::as_<Out> const&) const
    {
      typedef typename boost::simd::meta
                            ::vector_of < value_type
                                        , boost::simd::meta
                                               ::cardinal_of<Out>::value
                                        >::type in_tt;

       return nt2::run(data_ , in_position_ + p , meta::as_<in_tt>() );
    }


};
}}
#endif
