#include "backend.hpp"
#include "shared_handle.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <string>
#include <fstream>

std::string find_backend_module(const std::string& name)
{
  std::string path;
  path.resize(4096);
#ifdef BOOST_WINDOWS_API
  const char separator[] = "\\";
  GetModuleFileName(0, const_cast<char*>(path.data()), path.size());
#else
  const char separator[] = "/";
  readlink("/proc/self/exe", const_cast<char*>(path.data()), path.size());
#endif

  std::string::size_type pos = path.rfind(separator);
  if(pos != std::string::npos)
      path = path.substr( 0, pos );
  else
      path.clear();

  return path + separator + "backends" + separator + name + separator + name + shared_extension();
}

struct backend
{
  backend(const std::string& name)
         : handle(find_backend_module(name).c_str())
         , generate(handle.get<void(const char* filename, kernel_symbol const&)>("generate"))
  {
  }

  shared_handle handle;
  void (*generate)(const char* filename, kernel_symbol const&);
};

std::map<std::string, boost::shared_ptr<backend> > backends;

void launch_backend(const char* filename, kernel_symbol const& s)
{
  std::map<std::string, boost::shared_ptr<backend> >::iterator it = backends.find(s.target);
  if(it == backends.end())
    it = backends.insert(std::make_pair(s.target, boost::make_shared<backend>(s.target))).first;
  return it->second->generate(filename, s);
}
