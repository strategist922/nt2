//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_PGETRF_INCPIV_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_PGETRF_INCPIV_HPP_INCLUDED

#include <nt2/linalg/functions/pgetrf_incpiv.hpp>
#include <nt2/linalg/functions/gessm.hpp>
#include <nt2/linalg/functions/getrf_incpiv.hpp>
#include <nt2/linalg/functions/ssssm.hpp>
#include <nt2/linalg/functions/tstrf.hpp>

#include <nt2/linalg/details/plasma/grid.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <nt2/sdk/shared_memory.hpp>
#include <nt2/sdk/shared_memory/future.hpp>



namespace nt2 {

  namespace details
  {
    template <typename T1, typename T2>
    struct getrf_f
    {
        getrf_f(
        T1 & A_,
        std::size_t m_,
        std::size_t n_,
        std::size_t nb_,
        std::size_t ib_,
        std::size_t k_,
        T2 & IPIV_
        )
        :Aptr(& A_),m(m_),n(n_),nb(nb_),ib(ib_),k(k_)
        ,IPIVptr(& IPIV_)
        {}

      template<typename T>
      void operator()(T const &)
      {
        T1 & A(*Aptr);
        T2 & IPIV(*IPIVptr);

        nt2::getrf_incpiv(ib,
                          A(    _(k*nb+1,k*nb+m), _(k*nb+1,k*nb+n)),
                          IPIV( _(k*nb+1,k*nb+m), k)
                          );
      }

      T1 * Aptr;
      std::size_t m,n,nb,ib,k;
      T2 * IPIVptr;
    };

    template <typename T1, typename T2>
    struct tstrf_f
    {
       tstrf_f(
        T1 & A_,
        T1 & L_,
        std::size_t m_,
        std::size_t n_,
        std::size_t nb_,
        std::size_t ib_,
        std::size_t k_,
        std::size_t mm_,
        T2 & IPIV_
        )
      :Aptr(&A_),Lptr(&L_),m(m_),n(n_),nb(nb_),ib(ib_),k(k_),mm(mm_)
      ,IPIVptr(&IPIV_)
      {}

      template< typename T>
      void operator()(T const &)
      {
        T1 & A(*Aptr);
        T1 & L(*Lptr);
        T2 & IPIV(*IPIVptr);

        T1 work(nt2::of_size(m,n));

        nt2::tstrf(std::make_pair(ib,nb),
                   A(    _(k*nb+1, k*nb+nb),  _(k*nb+1,k*nb+n) ),
                   A(    _(mm*nb+1,mm*nb+m),  _(k*nb+1,k*nb+n) ),
                   L(    _(mm*ib+1,mm*ib+ib), _(k*nb*1,k*nb+n) ),
                   IPIV( _(mm*nb+1,mm*nb+m),  k                ),
                   work( _(1,m),              _(1,n)           )
                   );
      }

      double * Aptr;
      double * Lptr;
      std::size_t m,n,nb,ib,k,mm,info,LDA,LDL;
      std::size_t * IPIVptr;
    };

    template <typename T1, typename T2>
    struct gessm_f
    {
       gessm_f(
        T1 & A_,
        std::size_t m_,
        std::size_t n_,
        std::size_t nb_,
        std::size_t ib_,
        std::size_t k_,
        std::size_t nn_,
        T2 & IPIV_
        )
      :A(&A_),m(m_),n(n_),nb(nb_),ib(ib_),k(k_),nn(nn_)
      ,IPIVptr(IPIV_)
      {}

      template<typename T>
      void operator()(T const &)
      {
        T1 & A(*Aptr);
        T2 & IPIV(*IPIVptr);

        nt2::gessm(ib,
                   IPIV( _(k*nb+1, k*nb+m), k                   ),
                   A(    _(k*nb+1, k*nb+m), _(k*nb+1, k*nb+nb)  ),
                   A(    _(k*nb+1, k*nb+m), _(nn*nb+1, nn*nb+n) )
                   );
      }

      T1 * Aptr;
      std::size_t m,n,nb,ib,k,nn,LDA;
      T2 * IPIVptr;
    };

    template <typename T1, typename T2>
    struct ssssm_f
    {
       ssssm_f(
        T1 & A_,
        T1 & L_,
        std::size_t m_,
        std::size_t n_,
        std::size_t nb_,
        std::size_t ib_,
        std::size_t k_,
        std::size_t mm_,
        std::size_t nn_,
        T2 & IPIV_
        )
      :Aptr(&A_),Lptr(&L_),m(m_),n(n_),nb(nb_),ib(ib_),k(k_),mm(mm_),nn(nn_)
      ,IPIVptr(&IPIV_)
      {}

      template< typename T>
      void operator()(T const &)
      {
        T1 & A(*Aptr);
        T1 & L(*Lptr);
        T2 & IPIV(*IPIVptr);

       nt2::ssssm(ib,
                  A(    _(k*nb+1,k*nb+nb),   _(nn*nb+1,nn*nb+n) ),
                  A(    _(mm*nb+1,mm*nb+m),  _(nn*nb+1,nn*nb+n) ),
                  L(    _(mm*ib+1,mm*ib+ib), _(k*nb+1, k*nb+nb) ),
                  A(    _(mm*nb+1,mm*nb+m),  _(k*nb+1, k*nb+nb) ),
                  IPIV( _(mm*nb+1,mm*nb+m),  k                  )
                  );
      }

      T1 * A;
      T1 * L;
      std::size_t m,n,nb,ib,k,mm,nn;
      T2 * IPIV;
    };
  }

  namespace ext
  {
      NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::pgetrf_incpiv_, (nt2::tag::shared_memory_<Arch,Site>)
                                , (A0)(A1)(S1)(A2)(S2)(A3)(S3)(Arch)(Site)
                                , (scalar_< integer_<A0> >)
                                  ((container_< nt2::tag::table_, unspecified_<A1>, S1 >))
                                  ((container_< nt2::tag::table_, unspecified_<A2>, S2 >))
                                  ((container_< nt2::tag::table_, integer_<A3>, S3 >))
                                )
      {
       typedef void result_type;
       typedef typename nt2::make_future<Arch, std::size_t>::type Future;

       BOOST_FORCEINLINE result_type operator()( A0 const & nb, A1 & A, A2 & L, A3 & IPIV) const
       {
          std::size_t M = nt2::height(A);
          std::size_t N = nt2::width(A);
          std::size_t ib = (nb<40) ? nb : 40;

          std::size_t TILES = M/nb;
          std::size_t m = M/TILES;
          std::size_t n = N/TILES;

          std::size_t src(0), dst(1);

          std::vector< Grid<Future> > Tiles;
          Tiles.reserve(TILES+1);

          Tiles.push_back(Grid<Future>(TILES+1,TILES+1, nt2::make_ready_future<Arch,std::size_t>));

          for(std::size_t k=0; k <TILES; k++)
          Tiles.push_back(Grid<Future>(TILES-k,TILES-k));

          for(std::size_t k=0; k < TILES; k++) {

          std::size_t km = (k==TILES-1) ? M - k*m : m;
          std::size_t kn = (k==TILES-1) ? N - k*n : n;

          //step 1
          Tiles[dst](0,0) = Tiles[src](1,1).then(dgetrf_f<A2,A3>(A,km,kn,nb,ib,k,IPIV));
          //step 2
          for(std::size_t mm = k+1; mm < TILES; mm++) {

            std::size_t m_ = (mm==TILES-1) ? M -mm*m : m;

            Tiles[dst](mm-k,0) = when_all<Arch>(Tiles[dst](mm-k-1,0),
                                                Tiles[src](mm-k+1,1)
                                               )
                                 .then(dtstrf_f<A2,A3>(A,L,m_,kn,nb,ib,k,mm,IPIV));
          }

          //step 3
          for(std::size_t nn = k+1; nn < TILES; nn++) {

          std::size_t n_ = (nn==TILES-1) ? N - nn*n : n;

          Tiles[dst](0,nn-k) = when_all<Arch>(Tiles[dst](0,0),
                                              Tiles[src](1,nn-k+1)
                                             )
                              .then(dgessm_f<A2,A3>(A,km,n_,nb,ib,k,nn,IPIV));
          }

          //step 4
          for(std::size_t nn = k+1; nn < TILES; nn++) {
                  for(std::size_t mm=k+1; mm < TILES; mm++) {

                      std::size_t m_ = (mm==TILES-1) ? M - mm*m : m;
                      std::size_t n_ = (nn==TILES-1) ? N - nn*n : n;

                      Tiles[dst](mm-k,nn-k) = when_all<Arch>( Tiles[dst](mm-k,0),
                                                              Tiles[dst](mm-k-1,nn-k),
                                                              Tiles[src](mm-k+1,nn-k+1)
                                                            )
                                              .then(dssssm_f<A2,A3>(A,L,m_,n_,nb,ib,k,mm,nn,IPIV));
                  }
          }

          src ++;
          dst ++;
        }

      Tiles[src](0,0).get();
      }
    };
  }
}

#endif
