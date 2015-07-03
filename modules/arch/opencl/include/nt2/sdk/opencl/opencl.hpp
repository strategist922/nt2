#ifndef NT2_SDK_OPENCL_OPENCL_HPP_INCLUDED
#define NT2_SDK_OPENCL_OPENCL_HPP_INCLUDED

#include <boost/dispatch/functor/forward.hpp>
#include <boost/assert.hpp>

namespace nt2 { namespace memory
{
  template<class T> class opencl_buffer;
} }

namespace nt2 { namespace tag
{
  template<typename Site> struct opencl_ : Site
  {
    typedef void   device_tag;

    template<typename Container> struct device_traits
    {
      using value_type  = typename Container::declared_value_type;
      using buffer_type = memory::opencl_buffer<value_type>;
    };

    typedef Site  parent;
  };
} }

#ifdef NT2_HAS_OPENCL
BOOST_DISPATCH_COMBINE_SITE( nt2::tag::opencl_<tag::cpu_> )
#endif

#endif
