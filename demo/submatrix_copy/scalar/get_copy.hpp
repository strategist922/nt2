//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#ifndef GET_COPY_HPP_INCLUDED
#define GET_COPY_HPP_INCLUDED

#include <vector>
#include <iostream>

template< typename T>
void get_f( std::vector<T> const & f
          , std::vector<T> & fcopy
          , int nx
          , int ny
          , int i
          , int j
          )
{
    int dec = nx*ny;
    int ind1 = 0;
    int ind2 = i + j*nx;

    fcopy[ind2] = f[i + j*nx + ind1];
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i>0) ? f[(i-1) + j*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (j>0) ? f[i + (j-1)*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i<nx-1) ? f[(i+1) + j*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (j<ny-1)? f[i + (j+1)*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i>0 && j>0) ? f[(i-1) + (j-1)*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i<nx-1 && j>0) ? f[(i+1) + (j-1)*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i<nx-1 && j<ny-1) ? f[(i+1) + (j+1)*nx + ind1] : T(0.);
    ind1 += dec; ind2+=dec;

    fcopy[ind2] = (i>0 && j<ny-1) ? f[(i-1) + (j+1)*nx + ind1] : T(0.);
}


#endif
