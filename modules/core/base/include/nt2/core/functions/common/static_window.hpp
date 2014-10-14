#ifndef NT2_CORE_FUNCTIONS_COMMON_STATIC_WINDOW_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_STATIC_WINDOW_HPP_INCLUDED

#include <boost/array.hpp>
#include <boost/mpl/int.hpp>

#include <boost/simd/sdk/meta/iterate.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>

#include <nt2/include/functions/run.hpp>

namespace nt2 { namespace ext
{
  //////////////////////////////////////////////////////////////////////////////
  //STATIC WINDOW
  //////////////////////////////////////////////////////////////////////////////

  //STRUCTURES FOR SIMD MANAGEMENT

  template< int card_SIMD, int card_Stencil >
  struct is_SIMD_biggest
          : boost::mpl::bool_< card_SIMD  >= card_Stencil - 1 >
  {
  };

  //STRUCTURES FOR LOOP UNROLLING
  template< typename Data >
  struct static_update_loop
  {
    BOOST_FORCEINLINE
    static_update_loop( Data & data  )
                      : data_(data)
    {
    }

    template< int I >
    BOOST_FORCEINLINE
    void operator()() const
    {
      data_[I] = data_[I+1];
    }

    Data & data_;
  };

  template< typename T >
  struct array_helper
  {
    typedef T type;
  };


  template< typename Data, typename In_Val
            ,  typename In   >
  struct static_window_constructor_loop
  {
    BOOST_FORCEINLINE
    static_window_constructor_loop( Data & data , In const & in
                                  , size_t position_)
                                  : data_(data) , in_(in)
                                  , position(position_)
    {
    }

    template< size_t I >
    BOOST_FORCEINLINE
    void operator()()
    {
      size_t card = boost::simd::meta::cardinal_of<In_Val>::value;
       data_[I] =   nt2::run(in_ , position + I
                              , meta::as_< In_Val > ()
                              );
    }

    Data & data_;
    In const& in_;
    size_t position;
  };

  // MAIN CLASS
  template< typename In_Val , typename In , int st_size > class static_window
  {
    static_window& operator=(static_window const&);

    public :

    typedef typename In::value_type value_type;
    typedef boost::simd::meta::cardinal_of<In_Val> card_type;
    typedef typename boost::array< In_Val , st_size > register_array;
    std::size_t  in_position_;
    In const& data_ ;
    register_array data_array;

    //CONSTRUCTOR
    BOOST_FORCEINLINE
    static_window( In const& d, size_t begin_in )
                  : in_position_( begin_in  + st_size -1 )
                  , data_(d)
    {
      static_window_constructor_loop< register_array
                                    , In_Val , In
                                    > stepper( data_array
                                              , data_ , begin_in);
      boost::simd::meta::iterate<st_size -1 >(stepper);

    }


    //UPDATE
    BOOST_FORCEINLINE void update()
    {
      static_update_loop< register_array > stepper( data_array );
      boost::simd::meta::iterate<st_size-1>(stepper);
      in_position_ += card_type::value;
    }

    ///LOAD
    BOOST_FORCEINLINE void load() //LOAD HAVE TO BE CALLED AFTER UPDATE
    {
      data_array[st_size - 1 ] = nt2::run(data_ , in_position_
                                         , meta::as_< In_Val > ()
                                         );

    }

    //OPERATOR()
    BOOST_FORCEINLINE
    In_Val operator()(size_t p) const
    {
     return  data_array[p];
    }
  };

}}
#endif
