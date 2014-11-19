#ifndef NT2_SDK_MAGMA_FUTURE_FUTURE_HPP_INCLUDED
#define NT2_SDK_MAGMA_FUTURE_FUTURE_HPP_INCLUDED


#if defined(NT2_USE_MAGMA)

#include <type_traits>
#include <magma.h>
#include <cublas.h>
#include <cuda.h>
#include <nt2/sdk/magma/future/details/magma_future.hpp>

namespace nt2{


  namespace tag{
    template<class T> struct magma_;
  }

  template<class Arch, class result_type>
  struct make_future;

    template<class Site, class result_type>
  struct make_future<tag::magma_<Site> , result_type>
  {
    typedef details::magma_future<result_type> type;
  };

  template<class Arch>
  struct async_impl;

  template<class Arch, class F, class... ArgTypes>
  inline typename make_future< Arch
  , typename F::result_type
  >::type
  async(F && f, ArgTypes && ... args)
  {
    return async_impl<Arch>().call(std::forward<F>(f), std::forward<ArgTypes> (args)...);
  }

  template<class Site>
  struct async_impl< tag::magma_<Site> >
  {
    template<class F, class... ArgTypes>
    inline typename make_future< tag::magma_<Site>
    , typename F::result_type
    >::type
    call(F f , ArgTypes && ... args)
    {

      using result_type = typename F::result_type;
      using future = typename details::magma_future<result_type> ;

      future f1;
      f.call(f1.get_dres(), std::forward<ArgTypes>(args)...);

      return f1;
    }
  };
}

#endif
#endif
