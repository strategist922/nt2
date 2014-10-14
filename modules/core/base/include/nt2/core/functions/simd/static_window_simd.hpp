#ifndef NT2_CORE_FUNCTIONS_SIMD_STATIC_WINDOW_SIMD_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIMD_STATIC_WINDOW_SIMD_HPP_INCLUDED

#include <boost/array.hpp>

#include <boost/mpl/int.hpp>

#include <boost/simd/sdk/meta/iterate.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>

#include <nt2/include/functions/run.hpp>

namespace nt2 { namespace ext
{
  //////////////////////////////////////////////////////////////////////////////
  //STATIC WINDOW SIMD
  //////////////////////////////////////////////////////////////////////////////

  //STRUCTURES FOR LOOP UNROLLING
  template< typename Data >
  struct static_update_loop_simd
  {
    BOOST_FORCEINLINE
    static_update_loop_simd( Data & data  )
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


  template< typename Data, typename In_Val
            ,  typename In  , size_t card >
  struct static_window_constructor_loop_simd
  {
    BOOST_FORCEINLINE
    static_window_constructor_loop_simd( Data & data , In const & in
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
       data_[I] =   nt2::run(in_ , position + I*card
                              , meta::as_< In_Val > ()
                              );
    }

    Data & data_;
    In const& in_;
    size_t position;
  };

  // MAIN CLASS
  template< typename In_Val , typename In , int st_size > class static_window_simd
  {
    static_window_simd& operator=(static_window_simd const&);

    public :

    typedef typename In::value_type value_type;
    typedef boost::simd::meta::cardinal_of<In_Val> card_type;
    typedef boost::mpl::int_< st_size/ card_type::value + 2  > nb_registers;
    typedef typename boost::array< In_Val , nb_registers::value > register_array;

    std::size_t  in_position_;
    In const& data_ ;
    register_array data_array;

    //CONSTRUCTOR
    BOOST_FORCEINLINE
    static_window_simd( In const& d, size_t begin_in )
                  : in_position_( begin_in  )
                  , data_(d)
    {
      static_window_constructor_loop_simd< register_array
                                         , In_Val , In , card_type::value
                                          > stepper( data_array
                                              , data_ , begin_in);
      boost::simd::meta::iterate<nb_registers::value -1 >(stepper);
    }

    //SIMD UPDATE
    BOOST_FORCEINLINE void update()
    {
      static_update_loop_simd< register_array > stepper( data_array );
      boost::simd::meta::iterate<nb_registers::value - 1>(stepper);
      in_position_ += card_type::value;
    }

    ///LOAD
    BOOST_FORCEINLINE void load() //LOAD HAVE TO BE CALLED AFTER UPDATE
    {
      data_array[ nb_registers::value - 1 ] = nt2::run( data_
                                                        , in_position_
                                                          + card_type::value
                                                          * ( nb_registers::value - 1 )
                                                        , meta::as_< In_Val >()
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
