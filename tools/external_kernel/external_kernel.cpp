//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/config.hpp>
#include <boost/process.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/cstdint.hpp>
#include <filesystem/operations.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <ctime>

#include "parse_symbol.hpp"
#include "backend.hpp"

int base32_encode( const unsigned char *data, int length
                 , unsigned char *result, int bufSize)
{
  if (length < 0 || length > (1 << 28))
    return -1;

  int count = 0;
  if(length > 0)
  {
    int buffer = data[0];
    int next = 1;
    int bitsLeft = 8;
    while(count < bufSize && (bitsLeft > 0 || next < length))
    {
      if(bitsLeft < 5)
      {
        if(next < length)
        {
          buffer <<= 8;
          buffer |= data[next++] & 0xFF;
          bitsLeft += 8;
        }
        else
        {
          int pad = 5 - bitsLeft;
          buffer <<= pad;
          bitsLeft += pad;
        }
      }
      int index = 0x1F & (buffer >> (bitsLeft - 5));
      bitsLeft -= 5;
      result[count++] = "abcdefghijklmnopqrstuvwxyz234567"[index];
    }
  }
  if(count < bufSize)
    result[count] = '\000';
  return count;
}

boost::uint64_t hash_combine(boost::uint64_t s, boost::uint64_t b)
{
  return 2 * s + b;
}

std::string make_temporary_directory(const char* s , std::string const& tmpdir)
{
  boost::uint64_t h = 0;
  for(; *s; ++s)
    h = hash_combine(h, *s);

  std::string work_directory = tmpdir;

  char buffer[4096];
  int c = base32_encode( reinterpret_cast<unsigned char const*>(&h), sizeof(h)
                       , reinterpret_cast<unsigned char*>(buffer), sizeof(buffer)
                       );


  work_directory.insert(work_directory.end(), buffer, buffer+c);

  filesystem::create_directories(work_directory);
  return work_directory;
}

void link_files(std::string const& target, std::vector<std::string> const& files_to_link)
{
  // TODO
}

int main(int argc, char* argv[])
{
  bool using_cl = false;
  bool debug = false;
  bool display = false;
  bool out_dir = false;
  const char* generator = 0;
  int args_begin = 1;
  std::string tmpdir = "";

  std::srand(std::time(0));
  int kernel_number = std::rand();


  for(int i=1; i<argc; ++i)
  {

    if(!strcmp(argv[i], "--output-dir") && (i+1) < argc)
    {
      out_dir = true;
      ++i;
      tmpdir = argv[i];
      args_begin += 2;
      continue;
    }

    if(!strcmp(argv[i], "--using-cl"))
    {
      using_cl = true;
      ++args_begin;
      continue;
    }
    if(!strcmp(argv[i], "--debug"))
    {
      debug = true;
      ++args_begin;
      continue;
    }
    if(!strcmp(argv[i], "--display"))
    {
      display = true;
      ++args_begin;
    }
    if(!strcmp(argv[i], "--generator") && (i+1) < argc)
    {
      ++i;
      generator = argv[i];
      args_begin += 2;
    }
  }

  if (!out_dir) tmpdir = "/tmp/nt2_external_kernel/";

  if(args_begin >= argc)
  {
    std::cerr << "usage: external_kernel"
              <<   " [--using-cl | --debug | --display]"
              << " | [--generator <generator>]"
              << " <object file>"
              << std::endl;
    return 1;
  }

  namespace bp = boost::process;
  try
  {
    bp::context ctx;
    ctx.environment = bp::self::get_environment();

    bp::children cs;
    if(!using_cl)
    {
      std::string cppfilt;
      try
      {
        cppfilt = bp::find_executable_in_path("c++filt");
      }
      catch(...)
      {
      }

      std::vector<bp::pipeline_entry> entries;
      std::vector<std::string> args = boost::assign::list_of("nm")("-u");

      // if c++filt is not found, use built-in demangler
      if(cppfilt.empty())
      {
        args.push_back("-C");
        ctx.stdout_behavior = bp::capture_stream();
      }

      for(int i=args_begin; i<argc; ++i)
        args.push_back(argv[i]);
      entries.push_back(bp::pipeline_entry(bp::find_executable_in_path("nm"), args, ctx));

      if(!cppfilt.empty())
      {
        ctx.stdout_behavior = bp::capture_stream();
        args = {"c++filt"};
        entries.push_back(bp::pipeline_entry(bp::find_executable_in_path("c++filt"), args, ctx));
      }
      cs = bp::launch_pipeline(entries);
    }
    else
    {
      std::vector<bp::pipeline_entry> entries;
      std::vector<std::string> args = boost::assign::list_of("dumpbin")("/SYMBOLS");
      for(int i=args_begin; i<argc; ++i)
        args.push_back(argv[i]);
      entries.push_back(bp::pipeline_entry(bp::find_executable_in_path("dumpbin"), args, ctx));

      ctx.stdout_behavior = bp::capture_stream();
      std::vector<std::string> tmp = boost::assign::list_of("findstr")("UNDEF.*external_kernel");
      args = tmp;
      entries.push_back(bp::pipeline_entry(bp::find_executable_in_path("findstr"), args, ctx));

      cs = bp::launch_pipeline(entries);
    }

    ctx.stdout_behavior = bp::redirect_stream_to_stdout();
    std::string cmake_command = bp::find_executable_in_path("cmake");
    std::vector<std::string> args;
    std::vector<std::string> kernel_directories;

    bp::pistream &is = cs.back().get_stdout();
    std::string line;
    while(std::getline(is, line))
    {
      if(display)
        std::cout << "line = " << line << std::endl;

      kernel_symbol symbol;

      if(parse_symbol(using_cl, debug, line, symbol))
      {
        if(display)
          std::cout << symbol << std::endl;

        std::string work_directory = make_temporary_directory(line.c_str() , tmpdir);
        launch_backend(work_directory.c_str(), symbol, kernel_number);
        ++kernel_number;
        ctx.work_directory = work_directory;

        // configure
        args = {"cmake","."};

        if(generator)
        {
          args.push_back("-G");
          args.push_back(generator);
        }
        bp::status sc = bp::launch(cmake_command, args, ctx).wait();

        for (int i =0 ; i< args.size() ; ++i) {std::cout << "args : " << args[i] << std::endl;}

        std::cout << sc.exited() << "  " << sc.exit_status() << std::endl;

        if(!sc.exited() || sc.exit_status() != 0)
        {
          std::cerr << "configuring " << work_directory << " failed" << std::endl;
          continue;
        }

        // build
        std::vector<std::string> tmp = boost::assign::list_of("cmake")("--build")(".");
        args = tmp;
        bp::status sb = bp::launch(cmake_command, args, ctx).wait();
        if(!sb.exited() || sb.exit_status() != 0)
        {
          std::cerr << "building " << work_directory << " failed" << std::endl;
        }
        kernel_directories.push_back(work_directory);
      }
    }

    bp::status s = bp::wait_children(cs);
    if(!s.exited() || s.exit_status() != 0)
      std::cerr << "extracting symbols failed" << std::endl;

    std::vector<std::string> kernel_files;

    for(std::vector<std::string>::const_iterator it=kernel_directories.begin(); it!=kernel_directories.end(); ++it)
      kernel_files.push_back(*it + "/libkernel.a");
    link_files("libexternal_kernels.a", kernel_files);
    for(std::vector<std::string>::const_iterator it=kernel_directories.begin(); it!=kernel_directories.end(); ++it)
      filesystem::remove(it->c_str());
  }
  catch(std::exception const& e)
  {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "unknown error" << std::endl;
    return 1;
  }
}
