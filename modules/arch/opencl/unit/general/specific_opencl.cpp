//============================================================================== 
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI        
//                                                                               
//          Distributed under the Boost Software License, Version 1.0.           
//                 See accompanying file LICENSE.txt or copy at                  
//                     http://www.boost.org/LICENSE_1_0.txt                      
//============================================================================== 
#include <iostream>                                                              
                                                                                 
#include <nt2/table.hpp>                                                         
                                                                                 
#include <nt2/include/functions/zeros.hpp>                                       
#include <nt2/include/functions/ones.hpp>                                        
#include <nt2/include/functions/size.hpp>                                        
#include <nt2/core/functions/opencl/transform.hpp>                               
                                                                                 
#include <nt2/sdk/unit/tests/ulp.hpp>                                            
#include <nt2/sdk/unit/tests/relation.hpp>                                       
#include <nt2/sdk/unit/tests/basic.hpp>                                          
#include <nt2/sdk/unit/module.hpp>                                               
                                                                                 
#include <nt2/include/functions/is_equal.hpp>                                    
                                                                                 
#include <nt2/include/functions/log.hpp>                                         
#include <time.h>                                                                
                                                                                 
#include <nt2/sdk/opencl/settings/specific_data.hpp>
                                                                                 
namespace compute = boost::compute;                                              
                                                                                 
                                                                                 
NT2_TEST_CASE_TPL( direct_transform, (float) )                                   
{                                                                                
  nt2::table<T> S(nt2::of_size(5,5));

  boost::proto::value(S).specifics().allocate(3, 3, 9);

  compute::command_queue queue = compute::system::default_queue();
  compute::context context = queue.get_context();

  char source[] =
        "__kernel void dummy( __global float *A"
        "                       ) {"
        "       const uint my_x = get_global_id(0);"
        "       A[my_x] += 3;"
        "}";

  compute::program program = compute::program::create_with_source(source, context);
  program.build();

  compute::kernel kernel(program, "dummy");
  kernel.set_arg(0, boost::proto::value(S).specifics().data(2));


  NT2_TEST_EQUAL(1,1);
}
