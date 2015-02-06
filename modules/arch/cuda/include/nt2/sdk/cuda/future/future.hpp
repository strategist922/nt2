#ifndef NT2_SDK_CUDA_FUTURE_FUTURE_HPP_INCLUDED
#define NT2_SDK_CUDA_FUTURE_FUTURE_HPP_INCLUDED


#if defined(NT2_HAS_CUDA)

#include <type_traits>
#include <tuple>
#include <cublas.h>
#include <cuda.h>
#include <nt2/sdk/cuda/future/details/cuda_future.hpp>

namespace nt2{

  namespace tag{
    template<class T> struct cuda_;
  }

  template<class Arch, class result_type>
  struct make_future;

    template<class Site, class result_type>
  struct make_future<tag::cuda_<Site> , result_type>
  {
    typedef details::cuda_future<result_type> type;
  };

  template<class Arch>
  struct async_impl;

  template<class Arch, class F, class... ArgTypes>
  inline typename make_future< Arch
 // , typename std::result_of<F(ArgTypes...)>::type
 , typename F::type
  >::type
  async(F && f, ArgTypes && ... args)
  {
    return async_impl<Arch>().call(std::forward<F>(f), std::forward<ArgTypes> (args)...);
  }

  template<class Site>
  struct async_impl< tag::cuda_<Site> >
  {
    template<class F, class... ArgTypes>
    inline typename make_future< tag::cuda_<Site>
 // , typename std::result_of<F(ArgTypes...)>::type
   , typename F::type
    >::type
    call(F && f , ArgTypes && ... args)
    {
      using result_type = typename F::type;
      // using result_type = typename std::result_of<F(ArgTypes...)>::type;
      using future = typename details::cuda_future<result_type> ;

      // -> init future
      future f1;

      // get args for function object
      auto tup = std::make_tuple(args...);

      // asynchronous call to function object
      f(std::forward<ArgTypes>(args)... , 10);

      return f1;
    }
  };
}

#endif
#endif
