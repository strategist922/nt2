#ifndef NT2_CORE_FUNCTIONS_DETAILS_AS_STENCIL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_DETAILS_AS_STENCIL_HPP_INCLUDED

#include <boost/utility/enable_if.hpp>

#include <boost/mpl/bool.hpp>

#include <nt2/core/functions/common/static_stencil.hpp>
#include <nt2/core/functions/common/dynamic_stencil.hpp>

namespace nt2 { namespace details
{
  //STENCIL DISPATCHER
  template< typename T >
  struct  is_stencil_dynamic
        : boost::mpl::bool_<T::extent_type::static_status>
  {
  };

  //STENCIL DISPATCHER
  template<class T>
  BOOST_FORCEINLINE
  typename boost::enable_if< is_stencil_dynamic<T>
                            , ext::static_stencil<T>
                            >::type
  as_stencil(T const & t )
  {
    return ext::static_stencil<T>(t);
  }

  template<class T>
  BOOST_FORCEINLINE
  typename boost::disable_if< is_stencil_dynamic<T>
                            , ext::dynamic_stencil<T>
                            >::type
  as_stencil(T const & t )
  {
    return ext::dynamic_stencil<T>(t);
  }

}}
#endif
