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
#include <regex>
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
  std::cout << "USING OPENCL GENERATOR\n";

  std::string filename_output = std::string(filename) + "/generated.cpp";
  std::string filename_output_cu = std::string(filename) + "/generated_cl.cpp";
  std::vector<tagged_expression> v = symbol.arguments ;

  boost::format params = boost::format("");
  std::string params_call = "" ;
  std::string includes = "";
  std::map<std::string,std::string> map_includes;
  //TODO : change locality with enum type
  std::vector<std::size_t> locality;

  // define available functions for the back-end
  std::set<std::string> include_headers;

  //TODO: define opencl function headers
  if(symbol.target == "opencl")
    include_headers = opencl_fun_headers();

  std::cout << "--------------kernel-----------------\n";
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
  std::vector<std::vector<std::string> > fn_signatures;

  // recognize tie node to parse multiple expression separately
  if(rhs.operation == "tie")
  {
    rhs_expr.resize(rhs.children.size());
    for(std::size_t i = 0 ; i< rhs_expr.size() ; ++i)
    {
      rhs_expr[i] = boost::format("%1%(") % rhs.children[i].operation ;
      get_rhs(indx,rhs.children[i], rhs_expr[i] , params , params_call, locality);
      get_rhs_ops(indx,rhs.children[i], fn_signatures);
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
    get_rhs_ops(indx,rhs, fn_signatures);
    rhs_expr[0] = boost::format("%1%);") % rhs_expr[0] ;
    add_map_includes(rhs, map_includes , symbol.target );
  }

//  std::cout << "*****************************************\n";
//  std::cout << "fn_signatures = \n";
//  for(std::size_t i = 0 ; i < fn_signatures.size() ; ++i ) {
//    std::cout << fn_signatures[i][0] << "\n";
//    for(std::size_t j = 1 ; j < fn_signatures[i].size() ; ++j )
//      std::cout << fn_signatures[i][j] << "\t";
//    std::cout << "\n";
//  }
//  std::cout << "*****************************************\n";

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
  else core_expr = boost::format ("%1% = %2%") % lhs_expr[0] % rhs_expr[0] ;


  typedef std::map<std::string,std::string> mtype;
  typedef std::set<std::string> stype;

  // Add include headers for the kernel and modify function name if neccesary
  {
    std::string temp_expr = core_expr.str() ;

    includes = "#include <nt2/table.hpp>\n";
    for(mtype::iterator it = map_includes.begin() ; it != map_includes.end() ; ++it)
    {
      stype::const_iterator it_set = include_headers.find(it->first);
      // The function does not exist in compute::vector so add NT2 versions
      if( it_set == include_headers.end() )
        includes += it->second;
      else {
//TODO: Is this fully safe with an installed version of nt2?
//      Currently needed to have the correct tags for all functions
        includes =
          includes
          + "#include <nt2/include/functions/"
          + it->first
          + ".hpp>\n"
        ;
      }
    }
    core_expr = boost::format("%1%") % temp_expr;
  }

// TODO: possibly replace with boost::compute includes
  if(symbol.target == "opencl") {
    includes =
      includes
      + "#include <CL/cl.h>\n"
      + "#include <boost/compute/container/vector.hpp>\n"
      + "#include <string>\n"
      + "\n"
      + "namespace compute = boost::compute;\n"
      + "\n"
    ;
  }

  std::vector<std::string>  test_dummy(locality.size());
  std::size_t rhs_params_indx = 0;
  rhs_params(rhs,test_dummy,rhs_params_indx,locality.size()-lhs_size,0);

  for(std::size_t i =0 ; i < test_dummy.size() ; ++i)
  {
     boost::replace_all(test_dummy[i], "()" , "(a1)");
  }
//  std::cout << "---------------rhs params-----------------" << std::endl;
//  for(std::size_t i = 0; i < test_dummy.size() ; ++i)
//  {
//    std::cout << test_dummy[i] << std::endl;
//  }

//-----------------------write generated.cu file---------------------------//
  std::string kernl_indx ;

  if ( symbol.target == "opencl" )
    kernl_indx = "get_global_id(0)";

  std::string fn_sig = op_rhs + to_string(indx);

  boost::format cu_expr;

  std::string params_wrapper = boost::replace_all_copy(params.str() , "__restrict" , "" );
  std::string params_cukernel = boost::replace_all_copy(params_wrapper ,value_type , "" );
  boost::replace_all(params_cukernel ,"*" , "" );
  boost::replace_all(params_cukernel ,"const" , "" );
  boost::replace_all(params_cukernel ,"    " , " " );
  boost::replace_all(params_cukernel ,"   " , " " );
  boost::replace_all(params_cukernel ,"  " , " " );

  std::string kernel_wrapper = fn_sig + "_wrapper";


// generate kernel
  boost::format str_expr("");
  str_expr = boost::format("%1%std::string %2% ()\n{\n")
            % str_expr % fn_sig;
  str_expr = boost::format("%1%  std::string res(\"\");\n")
            % str_expr;

 // Add auxiliary function definitions
  for ( std::size_t i = 0 ; i < fn_signatures.size() ; ++i ) {
    stype::const_iterator it_set = include_headers.find(fn_signatures[i][1]);
    if( it_set == include_headers.end() ) {
      str_expr = boost::format("%1%  res += std::string(\"inline %2%\");\n")
                % str_expr % fn_signatures[i][0];
      str_expr = boost::format("%1%  res += nt2::opencl::%2%() + std::string(\"\\n\");\n")
                % str_expr % fn_signatures[i][1];
    }
  }

  std::regex compute_v_wrap0("(const)*\\s[a-zA-Z0-9]*\\*\\s");
  std::regex compute_core_exp("\\n");
  std::string test = std::regex_replace(core_expr.str(), compute_core_exp, "");
  std::string krnl_params = params_wrapper;
  krnl_params = std::regex_replace(krnl_params, compute_v_wrap0, " __global $& ");

  str_expr = boost::format("%1%  res += std::string(\"__kernel void %2% (%3%)\\n{\\n\");\n")
            % str_expr % fn_sig % krnl_params;
  str_expr = boost::format("%1%  res += std::string(\"  int %2% = %3%;\\n\");\n")
            % str_expr % rank % kernl_indx;
//std::cout << "CORE_EXPR = " << test << "\nEND_CORE_EXPR\n";
  str_expr = boost::format("%1%  res += std::string(\"  %2%\\n\");\n")
            % str_expr % test;
  str_expr = boost::format("%1%  res += std::string(\"}\\n\");\n")
            % str_expr;
  str_expr = boost::format("%1%\n  return res;\n}\n")
            % str_expr;

//  std::cout << "***********************************\n"
//            << "output = \n";
//  std::cout << fn_sig << "\n";
//  std::cout << str_expr
//            << "***********************************\n";

  std::string cl_params_wrapper = params_wrapper;
  std::regex compute_v_wrap1("\\s[a-zA-Z0-9]*\\*\\s");
  std::regex compute_v_wrap2("\\*");
  cl_params_wrapper = std::regex_replace(cl_params_wrapper, compute_v_wrap1, " compute::vector<$&> & ");
  cl_params_wrapper = std::regex_replace(cl_params_wrapper, compute_v_wrap2, "");

//TODO: insert kernel build/call to replace current form
  boost::format kernel_wrapper_fn;
  std::string kernel_wrapper_decl;
  kernel_wrapper_decl =
    + "void "
    + kernel_wrapper
    + "("
    + cl_params_wrapper
    + ", std::size_t dimGrid, std::size_t blockDim, "
    + "std::size_t gridNum, std::size_t blockNum, "
    + "compute::command_queue & queue)"
  ;

  kernel_wrapper_fn = boost::format("%1%%2%\n{\n")
    % kernel_wrapper_fn
    % kernel_wrapper_decl
  ;

// TODO: doesn't yet account for possibility of multiple contexts(devices)
  kernel_wrapper_fn = boost::format("%1%%2%%3%%4%%5%%6%")
    % kernel_wrapper_fn
    % "  compute::program program = \n"
    % "    compute::program::create_with_source("
    % (fn_sig + "().c_str()")
    % ", queue.get_context());\n"
    % "  program.build();\n\n"
  ;

  kernel_wrapper_fn = boost::format("%1%%2%%3%%4%")
    % kernel_wrapper_fn
    % "  compute::kernel kernel(program, \""
    % fn_sig
    % "\");\n"
  ;

//cl_params_wrapper = std::regex_replace(cl_params_wrapper, only_args, "$&");
  std::regex extract_vars("\\s[a-zA-Z0-9]*\\*\\s* (t[0-9]*)");
  std::smatch cl_fn_vars;
//  std::regex_search(params_wrapper, cl_fn_vars, extract_vars);
  std::sregex_iterator cl_it(params_wrapper.begin(), params_wrapper.end(), extract_vars);
  std::sregex_iterator regex_end;

{
  int i = 0;
  for ( auto it = cl_it ; it != regex_end ; ++it ) {
    kernel_wrapper_fn = boost::format("%1%%2%%3%%4%%5%%6%")
      % kernel_wrapper_fn
      % "  kernel.set_arg("
      % i
      % " , "
      % std::regex_replace( (*it).str(), extract_vars, "$1" )
      % ");\n"
    ;
    ++i;
  }
}

// TODO: Probably replace this with function inputs
  kernel_wrapper_fn = boost::format("%1%\n%2%%3%%4%%5%")
    % kernel_wrapper_fn
    % "  size_t dim = 1;\n"
    % "  size_t offset[] = { (dimGrid * gridNum) + (blockDim * blockNum) };\n"
    % "  size_t global_size[] = { dimGrid };\n"
    % "  size_t local_size[] = { blockDim };\n"
  ;

//TODO: If dim is limited to 1, you can simplify the enqueue call
  kernel_wrapper_fn = boost::format("%1%%2%")
    % kernel_wrapper_fn
    % "  queue.enqueue_nd_range_kernel(kernel, dim, offset, global_size, local_size);\n"
  ;

  kernel_wrapper_fn = boost::format("%1%%2%")
    % kernel_wrapper_fn
    % "\n}\n"
  ;

  cu_expr = boost::format("%1%\n%2%%3%")
    % includes
    % str_expr
    % kernel_wrapper_fn
  ;


  {
    std::ofstream output_cu(filename_output_cu.c_str(), std::ios::app);
    output_cu << cu_expr ;
  }

//--------------------------------streaming-----------------------------------//

  std::string stream_cu = "";

  stream_cu =
    stream_cu
    + "  std::size_t size = numel(boost::proto::child_c<0>(a1));\n"
    + "  std::size_t blockSize = std::min(std::size_t(256),size);\n"
    + "  std::size_t nQueues = std::min(std::size_t(4),size/blockSize);\n"
    + "  std::size_t leftover = size % blockSize ;\n"
    + "  std::size_t n = size / blockSize;\n"
//TODO: Check if you can make this change to n. If so, it simplifies the kernel loop
    + "//  std::size_t n = (size + blockSize - 1) / blockSize;\n"
    + "  std::size_t dimGrid  = blockSize;\n"
    + "  std::size_t blockDim = 1;\n"
    + "  compute::command_queue *queues = new compute::command_queue[nQueues];\n\n"
    + "  std::size_t spill;\n"
    + "  if ( leftover != 0 ) spill = 1;\n"
    + "  else spill = 0;\n"
    + "\n"
  ;

//  stream_cu =
//    stream_cu
//for ( std::size_t i = 0 ; i < nStreams ; ++i ) {
//  queues[i] = command::queue(compute::context(devices[i]), devices[i]);
//}
//    + "cudaStream_t stream[nQueues];\n"
//   + "  stream[j] = compute::command_queue(devices[i]);

  // Allocate device memory -- generated -- TODO : add informations to allocate
  // on is it an lhs ?
  // compute::vectors are already on device, aren't they?

//  stream_cu =
//    stream_cu
//    + "std::vector<compute::device> devices = compute::system::devices();\n"
//  ;

// TODO: Replace stream management with command queue management
  stream_cu =
    stream_cu
//   + "for(std::size_t i = 0; i < n ; ++i)\n"
//   + "{\n"
//   + "  std::size_t j = i % nQueues;\n"
//   + "}\n"
   ;


  // Detect all devides
//TODO: Move this before queue creation, make queue creation depend on devices
  std::string detect_device =
    "  std::vector<compute::device> devices;\n"
  ;

  detect_device =
    detect_device
    + "  for ( compute::device dev : compute::system::devices() )\n"
    + "    devices.push_back(dev);\n"
    + "\n"
  ;

  detect_device =
    detect_device
// TODO: Looks like you can create one context that uses all devices
    + "// TODO: take advantage of multiple devices\n"
    + "  std::vector<compute::context> contexts;\n"
    + "  contexts.resize(devices.size());\n"
    + "  for ( std::size_t i = 0 ; i < devices.size() ; ++i )\n"
    + "    contexts[i] = compute::context(devices[i]);\n"
  ;

  detect_device =
    detect_device
    + "  for ( std::size_t i = 0 ; i < nQueues ; ++i )\n"
//    + "    queues[i] = compute::command_queue(contexts[0], devices[0]);\n"
    + "    queues[i] = compute::command_queue(compute::system::default_context(), compute::system::default_device());\n"
  ;

  stream_cu =
    stream_cu
    + detect_device
    + "\n"
  ;

  stream_cu =
    stream_cu
    + "//------------ allocate necessary device memory ------------//\n"
  ;

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
            +  "  boost::proto::value(boost::proto::child_c<"
            + to_string(i)
            + ">(a0)).specifics().allocate(blockSize,nQueues,size,queues);\n"
          ;
        }
        else
        {
          stream_cu =
            stream_cu
            + "  boost::proto::value(a0).specifics().allocate(blockSize,nQueues,size,queues);\n"
          ;
        }
      }
    }
    else
    {
      if(locality[i] == 0)
      {
        stream_cu=
          stream_cu
          + "  boost::proto::value("
          + test_dummy[i-lhs_size]
          + ").specifics().allocate(blockSize,nQueues,size,queues);\n"
        ;
      }
    }
  }
  stream_cu =
    stream_cu
    + "\n"
  ;

  stream_cu =
    stream_cu
    + "   //------------ transfers host to device ------------//\n"
  ;

  stream_cu =
    stream_cu
    + "  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {\n"
    + "    std::size_t j = i % nQueues;\n"
    + "    std::size_t extra = 0;\n"
    + "    if ( i == n ) extra = leftover;\n"
  ;

// Transfer from CPU to GPU if necessary
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
            + "    boost::proto::value(boost::proto::child_c<"
            + k_s
            + ">(a0)).specifics().transfer_htd(boost::proto::child_c<"
            + k_s
            + ">(a0), i, queues["
            + "j"
            + "] , j );\n"
          ;
        }
        else
        {
          stream_cu =
            stream_cu
            + "    boost::proto::value(a0).specifics().transfer_htd(a0, i, queues["
            + "j"
            + "], j );\n"
          ;
        }
      }
      else
      {
        std::string num = to_string(k-lhs_size);
        stream_cu =
          stream_cu
          +  "    boost::proto::value("
          +  test_dummy[k-lhs_size]
          + ").specifics().transfer_htd("
          +  test_dummy[k-lhs_size]
          + ", i, queues["
          + "j"
          + "], j );\n"
        ;
      }
    }
  }
  stream_cu =
    stream_cu
    + "  }\n"
    + "\n"
  ;


//TODO: Check to make sure it's ok to call the kernel normally on the leftover
//        block. I'm 90% sure you don't segfault when going past allocated
//        memory on a GPU, but I can't find documenation to confirm...
  std::string kernel = "";

  kernel =
    kernel
    + "  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {\n"
    + "    std::size_t j = i % nQueues;\n"
  ;

  // write kernel --generated  -- add shift to data access if locality == 1
  std::string call_kernel = "    "+kernel_wrapper + "(\n" ;
  for(std::size_t k = 0 ; k < locality.size() ; k++)
  {
    std::string k_s = to_string(k);
    if(locality[k] == 2)
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          call_kernel += "      boost::proto::child_c<"+k_s+">(a0),\n";
        }
        else
        {
          call_kernel += "      a0,\n";
        }
      }
       else
        call_kernel += "      " + test_dummy[k-lhs_size] + ",\n";
    }

    else if(locality[k] == 1)
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          call_kernel += "      boost::proto::child_c<"+k_s+">(a0).data(),\n";
        }
        else
        {
          call_kernel += "      a0.data(),\n";
        }
      }
       else
        call_kernel += "      " + test_dummy[k-lhs_size] + ".data(),\n";
    }
    else
    {
      if( k < lhs_size)
      {
        if(lhs.operation =="tie")
        {
          call_kernel += "      boost::proto::value(boost::proto::child_c<"+k_s+">(a0)).specifics().data(j),\n";
        }
        else
        {
          call_kernel += "      boost::proto::value(a0).specifics().data(j),\n";
        }
      }
      else
        call_kernel += "      boost::proto::value("+test_dummy[k-lhs_size]+").specifics().data(j),\n";
    }
  }

  call_kernel += "      dimGrid, blockDim, 0, i, queues[j]\n";
  call_kernel += "    );\n";

  kernel += call_kernel;
  kernel += "  }// for i in n\n";

  stream_cu =
    stream_cu
    + "\n      //------------------ kernel call -------------------//\n\n"
    + kernel
  ;


  // Device -> Host -- generated -- only for lhs
  std::string dth_kernel = "";
  dth_kernel =
    dth_kernel
    + "  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {\n"
    + "    std::size_t j = i % nQueues;\n"
    + "    std::size_t extra = 0;\n"
    + "    if ( i == n ) extra = leftover;\n"
  ;

  for(std::size_t k = 0 ; k < lhs_size ; k++)
  {
   std::string k_s = to_string(k);
   if(locality[k] == 0)
    {
        if (lhs.operation =="tie")
        {
          dth_kernel =
            dth_kernel
            + "    boost::proto::value(boost::proto::child_c<"
            + k_s
            + ">(a0)).specifics().transfer_dth(boost::proto::child_c<"
            + k_s
            + ">(a0), n, queues[j], j , leftover );\n"
          ;
        }
        else
        {
          dth_kernel =
            dth_kernel
            + "    boost::proto::value(a0).specifics().transfer_dth(a0, n, queues[j], j, leftover );\n"
          ;
        }
    }
  }
  dth_kernel += "  }\n";

  stream_cu =
    stream_cu
    + "\n      //------------ transfers device to host ------------//\n\n"
    + dth_kernel
    + "\n"
    + "\n"
    + "  delete [] queues;\n"
    + "\n\n"
  ;

//-----------------------write generated.cpp file---------------------------//

  std::string kernel_trans =
    kernel_wrapper_decl
    + ";\n\n"
//    "void "
//    + kernel_wrapper
//    + "("
//    + params_wrapper
////    + ", std::size_t dimGrid, std::size_t blockDim, compute::command_queue & queue);\n\n"
//    + ", std::size_t dimGrid, std::size_t blockDim, "
//    + "std::size_t gridNum, std::size_t blockNum, "
//    + "compute::command_queue & queue);\n\n"
  ;

  std::string header_inc = "";
  header_inc =
    header_inc
    + "#include <nt2/sdk/external_kernel/external_kernel.hpp>\n"
    + "#include <nt2/sdk/"+symbol.target+"/"+symbol.target+".hpp>\n"
    + "#include <nt2/core/functions/transform.hpp>\n"
    + "#include <nt2/include/functions/height.hpp>\n"
    + "#include <nt2/include/functions/width.hpp>\n"
    + "#include <nt2/include/functions/numel.hpp>\n"
    + "#include <CL/cl.h>\n"
    + "#include <boost/compute/core.hpp>\n"
    + "#include <boost/compute/container/vector.hpp>\n"
    + "#include <nt2/sdk/opencl/settings/specific_data.hpp>\n"
  ;

  if(rhs.operation == "tie" )
    header_inc += "#include <nt2/include/functions/tie.hpp>\n";

  header_inc=
    header_inc
    + "\n"
    + "using boost::proto::child_c;\n"
    + "namespace compute = boost::compute;\n\n"
    + kernel_trans
    + "\n"
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
    + symbol.arguments[1].raw + "& a1)\n{\n"
  ;


  std::string params_call_raw = boost::replace_all_copy(params_call ,")" , ").data()" );

  transform_expr  =
    transform_expr
    + stream_cu
    + "} // kernel\n"
    + "} // namespace nt2\n"
  ;

  {
    std::ofstream output(filename_output.c_str(), std::ios::app);
    output << cu_expr
           << "\n\n//-----------------------------------------------\n\n"
           << transform_expr ;
  }

//----------------------------------------------------------------------------//
  std::cout << "---------------------kernel --------------------" << std::endl;
  std::cout << cu_expr << std::endl;

  std::cout << "---------------------host code------------------" << std::endl;
  std::cout << transform_expr << std::endl;

}

