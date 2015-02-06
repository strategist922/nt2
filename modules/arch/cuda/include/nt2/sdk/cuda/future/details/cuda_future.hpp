#ifndef NT2_SDK_CUDA_FUTURE_DETAILS_CUDA_FUTURE_HPP_INCLUDED
#define NT2_SDK_CUDA_FUTURE_DETAILS_CUDA_FUTURE_HPP_INCLUDED


#if defined(NT2_HAS_CUDA)

#include <memory>
#include <cublas.h>

namespace nt2{

  namespace tag{
    template<class T> struct cuda_;
  }

  namespace details{

    template<typename result_type>
    struct cuda_future
    {
      cuda_future() : res_( new result_type ),ready_(new bool(false))
      {
      }

      template<typename previous_future>
      void attach_previous_future(previous_future const & pfuture)
      {
        pfutures_.push_back(
          std::shared_ptr<previous_future>(
            new previous_future(pfuture)
            )
          );
      }

      inline bool is_ready() const
      {
        return *ready_;
      }


      inline void wait()
      {
        cudaDeviceSynchronize();
        ready_ = true;
      }

      inline result_type get()
      {
        if(!is_ready())
        {
          wait();
        }

        return *res_;
     }

      // template<typename F>
      // cuda_future<typename std::result_of<F(cuda_future)>::type>
      // then(F && f)
      // {

      //   using ftype = cuda_future<typename std::result_of<F(cuda_future)>::type> ;

      //   return f1 ;
      // }

      std::vector< std::shared_ptr<void> > pfutures_;
      std::shared_ptr<result_type> res_;
      std::shared_ptr<bool> ready_;
  };


}

}

#endif

#endif
