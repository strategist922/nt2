#ifndef NT2_SDK_MAGMA_FUTURE_DETAILS_MAGMA_FUTURE_HPP_INCLUDED
#define NT2_SDK_MAGMA_FUTURE_DETAILS_MAGMA_FUTURE_HPP_INCLUDED


#if defined(NT2_USE_MAGMA)

#include <memory>
#include <cublas.h>

namespace nt2{

  namespace tag{
    template<class T> struct magma_;
  }

  namespace details{

    template<typename result_type>
    struct magma_future
    {
      // may need a shared_ptr ?
      magma_future() : h_res( new result_type ), ready(false)
      {
        cudaMalloc( (void**) &d_res, sizeof(int)*10);
      }

      magma_future(result_type* r) : h_res( new result_type ), ready(false), d_res(r)
      {
      }

      inline bool is_ready() const
      {
        return ready;
      }

      inline result_type* get_dres() const
      {
        return d_res;
      }

      inline void wait()
      {
        cudaDeviceSynchronize();
        ready = true;
      }

      inline result_type* get()
      {
       if(!ready) wait();
       std::size_t size = sizeof(result_type)*10;
       cudaMemcpy(h_res, d_res , size, cudaMemcpyDeviceToHost );
       cudaFree(d_res);
       return h_res;
     }

    result_type * h_res;
    bool ready ;
    result_type * d_res;
  };


}

}

#endif

#endif
