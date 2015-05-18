#ifndef SHARED_HANDLE_HPP_INCLUDED
#define SHARED_HANDLE_HPP_INCLUDED

#include <boost/system/api_config.hpp>
#include <boost/throw_exception.hpp>
#include <boost/noncopyable.hpp>
#include <stdexcept>
#include <string>
#include <cstddef>

#ifdef BOOST_WINDOWS_API
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else // POSIX
    #include <dlfcn.h>
#endif

const char* shared_extension()
{
#ifdef BOOST_WINDOWS_API
  return ".dll";
#else
  return ".so";
#endif
}

struct shared_handle : boost::noncopyable
{
  shared_handle(const char* name)
  {
#ifdef BOOST_WINDOWS_API
    handle = ::LoadLibrary(name);
#else
    handle = ::dlopen(name, RTLD_LAZY);
#endif
    if(!handle)
      boost::throw_exception(std::runtime_error(std::string("Couldn't load library ") + name));
  }

  template<class T>
  T* get(const char* name)
  {
#ifdef BOOST_WINDOWS_API
    T* p = reinterpret_cast<T*>(::GetProcAddress(handle, name));
#else
    T* p = reinterpret_cast<T*>(reinterpret_cast<std::size_t>(::dlsym(handle, name)));
#endif
    if(!p)
      boost::throw_exception(std::runtime_error(std::string("Couldn't load symbol ") + name));

    return p;
  }

  ~shared_handle()
  {
#ifdef BOOST_WINDOWS_API
    ::FreeLibrary(handle);
#else
    ::dlclose(handle);
#endif
  }

private:
#ifdef BOOST_WINDOWS_API
  HMODULE handle;
#else
  void* handle;
#endif
};

#endif
