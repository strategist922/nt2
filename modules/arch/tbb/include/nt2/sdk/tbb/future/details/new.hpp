//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_NEW_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_NEW_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>
#include <new>
#include <cstdio>

  void * operator new (std::size_t size) throw (std::bad_alloc)
  {
      if(size==0) size = 1;
      if(void* ptr = scalable_malloc (size)) return ptr;
      throw std::bad_alloc();
  }

  void * operator new [] (std::size_t size) throw (std::bad_alloc)
  {
      return operator new(size);
  }

  void * operator new (std::size_t size, const std::nothrow_t &) throw ()
  {
      if(size==0) size = 1;
          if(void* ptr = scalable_malloc (size)) return ptr;
      return NULL;
  }

  void * operator new [] (std::size_t size, const std::nothrow_t &) throw ()
  {
      return operator new(size, std::nothrow);
  }

  void  operator delete (void * ptr) throw ()
  {
      if(ptr != 0) scalable_free(ptr);
  }

  void  operator delete [] (void * ptr) throw ()
  {
      operator delete(ptr);
  }

  void  operator delete (void * ptr, const std::nothrow_t &) throw ()
  {
      if(ptr != 0) scalable_free(ptr);
  }

  void  operator delete [] (void * ptr, const std::nothrow_t &) throw ()
  {
      operator delete(ptr, std::nothrow);
  }

 #endif
#endif
