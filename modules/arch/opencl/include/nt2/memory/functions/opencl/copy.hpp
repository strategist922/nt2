#ifndef NT2_MEMORY_FUNCTIONS_OPENCL_COPY_HPP_INCLUDED
#define NT2_MEMORY_FUNCTIONS_OPENCL_COPY_HPP_INCLUDED

#include <boost/dispatch/details/auto_decltype.hpp>

#if defined(NT2_HAS_OPENCL)

#include <nt2/sdk/memory/opencl/buffer.hpp>
#include <nt2/include/functions/size.hpp>
#include <boost/compute/container/vector.hpp>
#include <nt2/core/settings/locality.hpp>

namespace nt2 { namespace memory
{
namespace compute = boost::compute;

template<class T> class opencl_buffer;

template<typename Container, class T>
void copy(const opencl_buffer<T> & from, Container & into)
{
  if ( !from.size() ) return;
  if ( into.size() != from.size() )
    into.resize(from.size());

  compute::copy(from._vec.begin(), from._vec.end(), into.begin());
}

template<typename Container, class T>
void copy(const Container & from, opencl_buffer<T> & into)
{
  if ( !from.size() ) return;
  if ( into.size() != from.size() )
    into.resize(from.size());

  compute::copy(from.begin(), from.end(), into._vec.begin());
}

template<class T>
void copy(const opencl_buffer<T> & from, opencl_buffer<T> & into)
{
  if ( !from.size() ) return;
  if ( into.size() != from.size() )
    into.resize(from.size());

  compute::copy(from._vec.begin(), from._vec.end(), into._vec.begin());
}

template<class T>
void copy(opencl_buffer<T> & into , const opencl_buffer<T> & from)
{
  if ( !from.size() ) return;
  if ( into.size() != from.size() )
    into.resize(from.size());

  compute::copy(from._vec.begin(), from._vec.end(), into._vec.begin());
}

}} // end namespaces

#endif

#endif

//#ifndef NT2_MEMORY_FUNCTIONS_OPENCL_COPY_HPP_INCLUDED
//#define NT2_MEMORY_FUNCTIONS_OPENCL_COPY_HPP_INCLUDED
//
//#include <boost/dispatch/details/auto_decltype.hpp>
//
//#if defined(NT2_HAS_OPENCL)
//
//#include <nt2/sdk/memory/opencl/buffer.hpp>
//#include <nt2/include/functions/size.hpp>
//#include <boost/compute/container/vector.hpp>
//#include <nt2/core/settings/locality.hpp>
//
//namespace nt2 { namespace memory
//{
//namespace compute = boost::compute;
//
//template<class T> class opencl_buffer;
//
//template<typename Container, class T>
//void copy(Container & into, const opencl_buffer<T> & from)
//{
//  if ( !from.size() ) return;
//  if ( into.size() != from.size() )
//    into.resize(from.size());
//
//  compute::copy(from._vec.begin(), from._vec.end(), into.begin());
//}
//
//template<typename Container, class T>
//void copy(opencl_buffer<T> & into, const Container & from)
//{
//  if ( !from.size() ) return;
//  if ( into.size() != from.size() )
//    into.resize(from.size());
//
//  compute::copy(from.begin(), from.end(), into._vec.begin());
//}
//
//template<class T>
//void copy(opencl_buffer<T> & into, const opencl_buffer<T> & from)
//{
//  if ( !from.size() ) return;
//  if ( into.size() != from.size() )
//    into.resize(from.size());
//
//  compute::copy(from._vec.begin(), from._vec.end(), into._vec.begin());
//}
//
//}} // end namespaces
//
//#endif
//
//#endif
//
