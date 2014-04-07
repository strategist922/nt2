//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_WORKER_TRANSFORM_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_WORKER_TRANSFORM_HPP_INCLUDED

#include <nt2/sdk/shared_memory/worker.hpp>
#include <nt2/include/functor.hpp>

#include <utility>

namespace nt2
{

  namespace tag
  {
    struct transform_;
  }

  // Transform Worker
  template<class BackEnd,class Site, class Out, class In>
  struct worker<tag::transform_,BackEnd,Site,Out,In>
  {
      typedef int result_type;
      typedef typename boost::remove_reference<In>::type::extent_type extent_type;

      worker(Out const & out, In const & in)
      :out_(out),in_(in)
      {
         extent_type ext = in.extent();
         std::size_t bound  = boost::fusion::at_c<0>(ext);
      }

      int operator()(std::pair<std::size_t,std::size_t> begin
                    ,std::pair<std::size_t,std::size_t> size
                    ,std::size_t size_max)
      {

          for(std::size_t n=0; n<size.second; ++n)
          {
            std::size_t offset    = begin.first + (begin.second+n) * bound;

            std::size_t end = ( size_max < offset + size.first) )
            ? size_max % bound
            : size.first

            work(out_,in_,std::make_pair(offset,end));
          }

          return 0;
      };

      int operator()(int begin, int size)
      {
        work(out_,in_,std::make_pair(begin,size));
        return 0;
      };

      Out out_;
      In in_;
      std::size_t bound;
      nt2::functor<tag::transform_,Site> work;

  private:
  worker& operator=(worker const&);
  };

}
#endif
