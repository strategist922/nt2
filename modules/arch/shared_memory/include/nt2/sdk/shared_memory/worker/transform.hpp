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

#include <cstdio>
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
         extent_type ext = in_.extent();
         bound_  = boost::fusion::at_c<0>(ext);
      }

      int operator()(std::pair<std::size_t,std::size_t> begin // Indexes of the element in left up corner of Out tile
                    ,std::pair<std::size_t,std::size_t> chunk // height/width of Out tile
                    ,std::size_t offset                       // Container offset
                    ,std::size_t size)                        // Total number of elements to transform
      {
          for(std::size_t nn=0, n=begin.second; nn<chunk.second; ++nn, n+=bound_)
          {
            std::size_t o =  begin.first + n;

            if(size > o )
            {
              std::size_t colchunk = ( size < o + chunk.first)
              ? size - o
              : chunk.first;

              (*this)(offset + o, colchunk);
            }
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
      std::size_t bound_;
      nt2::functor<tag::transform_,Site> work;

  private:
  worker& operator=(worker const&);
  };

}
#endif
