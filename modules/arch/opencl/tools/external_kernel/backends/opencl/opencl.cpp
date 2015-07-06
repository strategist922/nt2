//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/config.hpp>
#include "types.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <boost/algorithm/string/replace.hpp>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <boost/format.hpp>
#include "utility.hpp"

template<class Expr>
void children_size(Expr const& rhs,std::size_t & res)
{

    for(std::size_t i = 0 ; i < rhs.children.size() ; ++i)
    {
        if(!is_terminal(rhs.children[i].operation))
            children_size(rhs.children[i],res );
        else ++res;

    }
}


template<class Expr>
void rhs_params(Expr const& rhs, std::vector<std::string> & params_call, std::size_t & indx, std::size_t const size_ , std::size_t par)
{
  for (std::size_t i = 0 ; i < rhs.children.size() ; ++i )
  {
    //  std::cout << "index is : " << indx << "  with children" <<  i << " op : " << rhs.children[i].operation << std::endl;
    if(is_terminal(rhs.children[i].operation))
    {
      params_call[indx] = "child_c<"+to_string(i)+">("+params_call[indx]+")";
      for(std::size_t k = 0 ; k < par ; ++k)
              params_call[indx] += ")";
     ++indx;
    }
    else // add the function and call get_rhs recursively
    {
        std::size_t child_size = 0;
        children_size(rhs.children[i],child_size);
        std::cout << "in else indx : " << indx << "  children size " << child_size << std::endl;
      for(std::size_t j = indx; j < indx + child_size ; ++j)
      {
          params_call[j] = "child_c<"+to_string(i)+">(" + params_call[j];
      }

      rhs_params(rhs.children[i], params_call, indx, size_,par+1);

    }
  }
}




extern "C" BOOST_SYMBOL_EXPORT void generate(const char* filename, kernel_symbol const& symbol)
{
  std::string filename_output = std::string(filename) + "/generated.cpp";
  std::string filename_output_cu = std::string(filename) + "/generated_cl.cl";
  std::vector<tagged_expression> v = symbol.arguments ;

  boost::format params = boost::format("");
  std::string params_call = "" ;
  std::string includes = "";
  std::map<std::string,std::string> map_includes;
  //TODO : change locality with enum type
  std::vector<std::size_t> locality;

  // define available functions for the back-end
  std::set<std::string> include_headers;

  if(symbol.target == "opencl")
    include_headers = cuda_fun_headers();

  std::cout << "---------------kernel-----------------" << std::endl;
  std::cout << symbol.kernel << std::endl;
  std::cout << symbol.target << std::endl;

//--------------------------Settings left-hand side---------------------------//
  expression lhs = v[0].expr;
  std::string value_type = get_value_type(lhs);
  display_info(lhs);

  std::vector<boost::format> lhs_expr(lhs.children.size(), boost::format("") );

  // parse lhs expression and recognize tie node for multiple outputs
  if(lhs.operation =="tie")
  {
    lhs_expr.resize(lhs.children.size());

    for(std::size_t i = 0; i < lhs_expr.size() ; ++i)
    {
      get_lhs(i,lhs.children[i],lhs_expr[i],params, value_type, locality);
      params_call += "boost::proto::child_c<"+to_string(i)+">(a0),";
    }
    params_call.erase(params_call.size()-1);
  }
  else
  {
    lhs_expr.resize(1);
    get_lhs(0,lhs, lhs_expr[0], params, value_type, locality);
    params_call += "boost::proto::child_c<0>(a0)";
  }

//--------------------------Settings right-hand side--------------------------//

  expression rhs = v[1].expr;
  std::string op_rhs = rhs.operation ;
  std::vector<boost::format> rhs_expr(rhs.children.size(), boost::format("") );

  display_info(rhs);

  // define first parameter number for rhs
  int indx = locality.size();

  // keep the number of params for lhs
  int lhs_size = indx;

  // recognize tie node to parse multiple expression separately
  if(rhs.operation == "tie")
  {
    rhs_expr.resize(rhs.children.size());
    for(std::size_t i = 0 ; i< rhs_expr.size() ; ++i)
    {
      rhs_expr[i] = boost::format("%1%(") % rhs.children[i].operation ;
      get_rhs(indx,rhs.children[i], rhs_expr[i] , params , params_call, locality);
      rhs_expr[i] = boost::format("%1%);") % rhs_expr[i] ;
      add_map_includes(rhs.children[i], map_includes , symbol.target );
    }
  }
  // Do we really want to call external kernel for table<T> a0 = table<T> a1 ?
  else if( is_terminal(rhs.operation) )
  {
    rhs_expr.resize(1);
    std::string index = "t" + to_string(indx) ;
    if(rhs.type.container)
    {
      is_on_device(rhs, locality);
      rhs_expr[0] = boost::format("%1%[%2%];") % index % rank;
      params = boost::format ("%1%, %2%* %3%") % params % value_type % index ;
    }
    else
    {
      locality.push_back(2);
      rhs_expr[0] = boost::format("%1%;") % index ;
      params = boost::format ("%1%, %2% %3%") % params % value_type % index ;
    }
    params_call += ",boost::proto::child_c<0>(a1)";
  }
  else
  {
    rhs_expr.resize(1);
    rhs_expr[0] = boost::format("%1%(") % op_rhs ;
    get_rhs(indx,rhs, rhs_expr[0] , params , params_call, locality);
    rhs_expr[0] = boost::format("%1%);") % rhs_expr[0] ;
    add_map_includes(rhs, map_includes , symbol.target );
  }

  // parameter call for the right-hand side
  for(std::size_t i = 0 ; i < indx - lhs_size ; ++i )
  {
    params_call += ",boost::proto::child_c<"+to_string(i)+">(a1)";

  }

  // recursive parameter call
  {
  }

  boost::format core_expr = boost::format("") ;

//------------------------------write Kernel ---------------------------------//

  // Generate core_expr for expressions including tie
  if (rhs.operation == "tie")
  {
    for(std::size_t i = 0 ; i< rhs_expr.size() ; ++i)
    {
      core_expr = boost::format ("%1% %2% = %3%%4%") % core_expr % lhs_expr[i]
                                                     % rhs_expr[i] % "\n";
    }
  }
  else core_expr = boost::format ("%1% = %2%%3%") % lhs_expr[0] % rhs_expr[0] % "\n";

  typedef std::map<std::string,std::string> mtype;
  typedef std::set<std::string> stype;

  // Add include headers for the kernel and modify function name if neccesary
  {
    std::string temp_expr = core_expr.str() ;

    for(mtype::iterator it = map_includes.begin() ; it != map_includes.end() ; ++it)
    {
      stype::const_iterator it_set = include_headers.find(it->first);
      // The function does not exist in compute::vector so add NT2 versions
      if( it_set == include_headers.end() )
        includes += it->second;
    }
    core_expr = boost::format("%1%") % temp_expr;
  }

// TODO: possibly replace with boost::compute includes
  if(symbol.target == "opencl")
    includes += "#include <CL/cl.h>\n";

  std::vector<std::string>  test_dummy(locality.size());
  std::size_t rhs_params_indx = 0;
   rhs_params(rhs,test_dummy,rhs_params_indx,locality.size()-lhs_size,0);

    for(std::size_t i =0 ; i < test_dummy.size() ; ++i)
    {
     boost::replace_all(test_dummy[i], "()" , "(a1)");
    }
   std::cout << "---------------rhs params-----------------" << std::endl;
  for(std::size_t i = 0; i < test_dummy.size() ; ++i)
  {
    std::cout << test_dummy[i] << std::endl;
  }

//-----------------------write generated.cu file---------------------------//
  std::string kernl_indx ;

// opencl directly has index
    kernl_indx = "get_global_id(0)";
//  if(symbol.target == "cuda")
//    kernl_indx = "blockIdx.x*blockDim.x+threadIdx.x";

  boost::format expr ("__global__ void %1%%2% (%3%)\n{\n int %4% = %5%;\n%6%}\n");
  expr = expr % op_rhs % to_string(indx) % params % rank % kernl_indx % core_expr;

  boost::format cu_expr = boost::format("%1%\n %2%") % includes % expr ;

  std::string params_wrapper = boost::replace_all_copy(params.str() , "__restrict" , "" );
  std::string params_cukernel = boost::replace_all_copy(params_wrapper ,value_type , "" );
  boost::replace_all(params_cukernel ,"*" , "" );
  boost::replace_all(params_cukernel ,"const" , "" );
  boost::replace_all(params_cukernel ,"    " , " " );
  boost::replace_all(params_cukernel ,"   " , " " );
  boost::replace_all(params_cukernel ,"  " , " " );

  std::string kernel_wrapper = op_rhs + to_string(indx) + "_wrapper";

// TODO: Is this a declaration or function call? Needs modification, but extent unknown
  cu_expr = boost::format("%1%\n%2%%3%%4%")
  % cu_expr
  % ("void " + kernel_wrapper + "(")
  % params_wrapper
  % (", dim3 dimGrid, dim3 blockDim, cudaStream_t stream = 0)\n{\n"
    + op_rhs + to_string(indx) + "<<<dimGrid,blockDim,0,stream>>>" + "(" + params_cukernel + ");\n}\n"
    ) ;

  {
    std::ofstream output_cu(filename_output_cu.c_str(), std::ios::app);
    output_cu << cu_expr ;
  }


//--------------------------------streaming-----------------------------------//

  std::string stream_cu = "";

 stream_cu =
  stream_cu
  + "std::size_t size = numel(boost::proto::child_c<0>(a1));\n"
  + "std::size_t blockSize = std::min(std::size_t(256),size);\n"
  + "std::size_t nStreams = std::min(std::size_t(4),size/blockSize);\n"
  + "std::size_t leftover = size % blockSize ;\n"
  + "std::size_t n = size / blockSize;\n"
  + "dim3 dimGrid  = blockSize;\n"
  + "dim3 blockDim = 1;\n"
  + "cudaStream_t stream[nStreams];\n"
  + "\n\n"
  ;

  // Allocate device memory -- generated -- TODO : add informations to allocate
  // on is it an lhs ?
  for(std::size_t i = 0; i<locality.size() ; ++i )
  {
    if(i < lhs_size )
    {
      if(locality[i] == 0)
      {
        if(lhs.operation =="tie")
        {
        stream_cu =
          stream_cu
          + "boost::proto::value(boost::proto::child_c<"
          +to_string(i)
          +">(a0)).specifics().allocate(blockSize,nStreams,size);\n";
        }
        else
        {
          stream_cu =
          stream_cu
          + "boost::proto::value(a0).specifics().allocate(blockSize,nStreams,size,true);\n";
        }
      }
    }
    else
    {
      if(locality[i] == 0)
      {
        stream_cu=
          stream_cu
          +"boost::proto::value("
          +test_dummy[i-lhs_size]
          +").specifics().allocate(blockSize,nStreams,size);\n";
      }
    }
  }

// TODO: Replace stream management with command queue management
  stream_cu =
    stream_cu
   + "\nfor(std::size_t i=0; i < nStreams ; ++i)\n"
   + "{\n"
   + "   cudaStreamCreate(&stream[i]);\n"
   + "}\n\n"
   + "for(std::size_t i = 0; i < n ; ++i)\n"
   + "{\n"
   + "   std::size_t j = i % nStreams;\n"
   +
   + "   //------------ transfers host to device ------------//\n\n"
   ;

  // Host -> device -- generated
  for(std::size_t k = 0 ; k < locality.size() ; k++)
  {
    std::string k_s = to_string(k);
    if(locality[k] == 0 )
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
        stream_cu =
                  stream_cu
                  +"   boost::proto::value(boost::proto::child_c<"
                  +k_s
                  +">(a0)).specifics().transfer_htd(boost::proto::child_c<"
                  +k_s
                  +">(a0), i, stream["
                  +"j"
                  +"] , j );\n"
                  ;
        }
        else
        {
        stream_cu =
                  stream_cu
                  +"   boost::proto::value(a0).specifics().transfer_htd(a0, i, stream["
                  +"j"
                  +"], j );\n"
                  ;
        }
      }
      else
      {
        std::string num = to_string(k-lhs_size);
        stream_cu += "   boost::proto::value("
                  + test_dummy[k-lhs_size]
                  +").specifics().transfer_htd("
                  + test_dummy[k-lhs_size]
                  +", i, stream["
                  +"j"
                  +"], j );\n";
      }
    }
  }

  // write kernel --generated  -- add shift to data access if locality == 1
  std::string kernel = "      "+kernel_wrapper + "(" ;
  for(std::size_t k = 0 ; k < locality.size() ; k++)
  {
    std::string k_s = to_string(k);
    if(locality[k] == 2)
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          kernel += " boost::proto::child_c<"+k_s+">(a0),";
        }
        else
        {
          kernel += " a0,";
        }
      }
       else
        kernel += test_dummy[k-lhs_size]+",";
    }

    else if(locality[k] == 1)
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          kernel += " boost::proto::child_c<"+k_s+">(a0).data(),";
        }
        else
        {
          kernel += " a0.data(),";
        }
      }
       else
        kernel += test_dummy[k-lhs_size]+".data(),";
    }
    else
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          kernel += " boost::proto::value(boost::proto::child_c<"+k_s+">(a0)).specifics().data(j),";
        }
        else
        {
          kernel += " boost::proto::value(a0).specifics().data(j),";
        }
      }
      else
        kernel += " boost::proto::value("+test_dummy[k-lhs_size]+").specifics().data(j),";
    }
  }
  kernel.erase(kernel.size()-1);
  kernel +=",dimGrid,blockDim,stream[j]);\n";

  stream_cu =
      stream_cu
      + "\n      //------------------ kernel call -------------------//\n\n"
      + kernel
      + "\n      //------------ transfers device to host ------------//\n\n"
      ;

  // Device -> Host -- generated -- only for lhs
  for(std::size_t k = 0 ; k < lhs_size ; k++)
  {
   std::string k_s = to_string(k);
   if(locality[k] == 0)
    {
        if (lhs.operation =="tie")
        {
          stream_cu =
          stream_cu
          +"   boost::proto::value(boost::proto::child_c<"
          +k_s
          +">(a0)).specifics().transfer_dth(boost::proto::child_c<"
          +k_s
          +">(a0), i,stream[j], j);\n}\n";
        }
        else
        {
          stream_cu =
          stream_cu
          +"   boost::proto::value(a0).specifics().transfer_dth(a0, i, stream[j], j );\n"
          // +"}\n"
          ;
        }
    }
  }

// LAST iteration if blocksize not multiple of data size --TODO write a function for each part
stream_cu += "\nif(leftover != 0)\n{\n   std::size_t j = n % nStreams;\n   dimGrid = leftover;\n\n   //------------ transfers host to device ------------//\n\n";

  // Host -> device -- generated
  for(std::size_t k = 0 ; k < locality.size() ; k++)
  {
    std::string k_s = to_string(k);
    if(locality[k] == 0 )
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
        stream_cu =
                  stream_cu
                  +"   boost::proto::value(boost::proto::child_c<"
                  +k_s
                  +">(a0)).specifics().transfer_htd(boost::proto::child_c<"
                  +k_s
                  +">(a0), n, stream[j], j, leftover"
                  +");\n"
                  ;
        }
        else
        {
        stream_cu =
                  stream_cu
                  +"   boost::proto::value(a0).specifics().transfer_htd(a0, n, stream[j], j ,leftover);\n"
                  ;
        }
      }
      else
      {
        std::string num = to_string(k-lhs_size);
        stream_cu += "   boost::proto::value("
                  +test_dummy[k-lhs_size]
                  +").specifics().transfer_htd("
                  +test_dummy[k-lhs_size]
                  +", n,stream[j], j , leftover );\n";
      }
    }
  }

  stream_cu =
      stream_cu
      + "\n      //------------------ kernel call -------------------//\n\n"
      + kernel
      + "\n      //------------ transfers device to host ------------//\n\n"
      ;

  // Device -> Host -- generated -- only for lhs
  for(std::size_t k = 0 ; k < lhs_size ; k++)
  {
   std::string k_s = to_string(k);
   if(locality[k] == 0)
    {
        if (lhs.operation =="tie")
        {
          stream_cu =
          stream_cu
          +"   boost::proto::value(boost::proto::child_c<"
          +k_s
          +">(a0)).specifics().transfer_dth(boost::proto::child_c<"
          +k_s
          +">(a0), n, stream[j], j , leftover );\n";
        }
        else
        {
          stream_cu =
          stream_cu
          +"   boost::proto::value(a0).specifics().transfer_dth(a0, n, stream[j], j, leftover );\n"

          ;
        }
    }
  }

stream_cu =
  stream_cu
  + "}\n\nfor(std::size_t i=0; i < nStreams ; ++i)\n"
  + "{\n"
  + "   cudaStreamDestroy(stream[i]);\n"
  + "}\n\n"
  ;

//-----------------------write generated.cpp file---------------------------//

 std::string kernel_trans = "void " + kernel_wrapper + "(" + params_wrapper
   + ", dim3 dimGrid, dim3 blockDim, cudaStream_t stream = 0);\n\n";

  std::string header_inc = "";
  header_inc =
  header_inc
  + "#include <nt2/sdk/external_kernel/external_kernel.hpp>\n"
  + "#include <nt2/sdk/"+symbol.target+"/"+symbol.target+".hpp>\n"
  + "#include <nt2/core/functions/transform.hpp>\n"
  + "#include <nt2/include/functions/height.hpp>\n"
  + "#include <nt2/include/functions/width.hpp>\n"
  + "#include <nt2/include/functions/numel.hpp>\n"
  + "#include <cuda.h>\n"
  ;

  if(rhs.operation == "tie" ) header_inc += "#include <nt2/include/functions/tie.hpp>\n";

  header_inc=
  header_inc
  + "\n\n"
  + kernel_trans
  + "using boost::proto::child_c;\n\n"
  ;

  std::string transform_expr =
  header_inc
  + "namespace nt2 {\n\n"
  + "template<> template <>\n"
  + "void nt2::external_kernel<"
  + "nt2::tag::" + symbol.kernel + "_, "
  + "nt2::tag::" + symbol.target + "_" + symbol.archi_tag + " >::call<"
  + symbol.arguments[0].raw + ", "
  + symbol.arguments[1].raw + "> ("
    + symbol.arguments[0].raw + "& a0, "
    + symbol.arguments[1].raw + "& a1)\n{\n";


  std::string dimGrid = "int dimGrid = 1;\n";
  std::string blockDim = "int blockDim = 128;\n";
  std::string params_call_raw = boost::replace_all_copy(params_call ,")" , ").data()" );

  transform_expr  =
    transform_expr
  + stream_cu
  // + dimGrid
  // + blockDim
  // + op_rhs + to_string(indx) + "_wrapper" + "(" + params_call_raw
  // + ", dimGrid, blockDim" + ");\n"
  + "}\n}\n";

  {
    std::ofstream output(filename_output.c_str(), std::ios::app);
    output << transform_expr ;
  }

//----------------------------------------------------------------------------//
  std::cout << "---------------------kernel --------------------" << std::endl;
  std::cout << cu_expr << std::endl;

  std::cout << "---------------------host code------------------" << std::endl;
  std::cout << transform_expr << std::endl;

}

