#ifndef NT2_SDK_RESOURCE_HPP
#define NT2_SDK_RESOURCE_HPP

#ifdef __DEPRECATED
#undef __DEPRECATED
#include <strstream>
#define __DEPRECATED
#else
#include <strstream>
#endif
#include <sstream>
#include <map>
#include <memory>
#include <string>

#include <boost/assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/iterator_range.hpp>

#define NT2_STREAM_STRING(StreamExpr) static_cast<std::ostringstream&>(std::ostringstream() << StreamExpr).str().c_str()

namespace nt2
{
  namespace detail
  {
    template<class T = void>
    struct resource_map
    {
      typedef std::map<std::string, boost::iterator_range<const char*> > type;
      static type value;
    };

    template<class T>
    resource_map<>::type resource_map<T>::value;

    struct register_handle
    {
      register_handle(const char* str, const char* data, std::size_t size)
      {
        BOOST_ASSERT_MSG(resource_map<>::value.find(str) == resource_map<>::value.end(), NT2_STREAM_STRING("Resource " << str << " already exists"));
        resource_map<>::value.insert(std::make_pair(str, boost::make_iterator_range(data, data + size)));
      }
    };
}

#define NT2_RESOURCE_REGISTER(str, data) static nt2::detail::register_handle nt2_detail_register_handle_(str, data, sizeof data);

  typedef detail::resource_map<>::type::const_iterator resource_iterator;

  inline resource_iterator resource_begin()
  {
    return detail::resource_map<>::value.begin();
  }

  inline resource_iterator resource_end()
  {
    return detail::resource_map<>::value.end();
  }

  inline boost::shared_ptr<std::istream> resource_stream(boost::iterator_range<const char*> const& data)
  {
    return boost::make_shared<std::istrstream>(data.begin(), data.end() - data.begin());
  }

  inline boost::shared_ptr<std::istream> get_resource(const char* name)
  {
    BOOST_ASSERT_MSG( detail::resource_map<>::value.find(name) != detail::resource_map<>::value.end()
                     , NT2_STREAM_STRING("Resource " << name << " does not exist")
                     );
    boost::iterator_range<const char*> data = detail::resource_map<>::value.find(name)->second;
    return resource_stream(data);
  }
}

#endif
