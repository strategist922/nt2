#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

  std::string bflags = "/usr/bin/cc -fno-strict-aliasing -DBOOST_SIMD_NO_STRICT_ALIASING -Wall -Wno-delete-non-virtual-dtor -Wno-array-bounds -Wextra -Wshadow -std=c++11 -DNT2_HAS_CUDA ";

  std::string boost = std::getenv("BOOST_ROOT");
  std::string nt2   = std::getenv("NT2_ROOT");

int main(int argc , char** argv)
{

    std::vector<std::string> args( argv +3 ,argv +argc);

    std::string fname(argv[1]);
    std::string foutput(argv[2]);
    std::string gpuflags = "";

    for(auto & s : args)
     {
      auto pos = s.find("-DDEVICE=");
      if ( pos != std::string::npos)
        {
           gpuflags += " " + s.substr(pos+9);
        }
      else {bflags += " " + s;}

     }

     bflags += " -I" + boost + " -lcuda -lcublas -lcudart -lnt2 -lstdc++ ";

     std::string flags = bflags + " -c " + fname +  " -o "+ fname + ".o";

     std::cout << flags << std::endl << std::endl;

     std::system(  (flags ).c_str() );

     // std::string external_kernel = nt2 + "/share/nt2/tools/external_kernel/external_kernel " + fname + ".o > stdout.txt 2> stderr.txt";
     std::string external_kernel = "/home/imasliah/dev/nt2_cleanext/build/tools/external_kernel/external_kernel " + fname + ".o > stdout.txt 2> stderr.txt";

     std::cout << external_kernel <<std::endl << std::endl;

     std::system( external_kernel.c_str() );

    std::ifstream ifs("stderr.txt");

      std::string line;
      int filenum = 0;

      while( std::getline(ifs, line))
      {
        auto pos = line.find('/');
        line = line.substr(pos);
        pos = line.find(' ');
        line = line.substr(0,pos);

        std::cout << line << std::endl;

        std::string flags = bflags + " -c " + line + "/generated.cpp  -o generated" + std::to_string(filenum) + ".cpp.o";

        std::string cudaflags = "nvcc " + gpuflags + " -I " + nt2 + "/include/nt2 -c " + line + "/generated_cu.cu -o generated_cu" + std::to_string(filenum) + ".o";

        std::cout << flags << std::endl <<std::endl;
        std::cout << cudaflags << std::endl<<std::endl;
        std::system( flags.c_str() );
        std::system( cudaflags.c_str() );
        ++filenum;
      }


      std::string gen= " ";

      for (size_t i = 0 ; i < filenum ; ++i)
      {
        gen += " generated_cu" + std::to_string(i) + ".o " + " generated" + std::to_string(i) + ".cpp.o";
      }


      flags = bflags + fname + gen + " -o " + foutput;

      std::cout << flags << std::endl << std::endl;

      std::system( flags.c_str());

// cleanup
      for (size_t i = 0 ; i < filenum ; ++i)
      {
        std::string clean = "rm generated" + std::to_string(i) + ".cpp.o";
        std::system( clean.c_str() );
        clean = "rm generated_cu" + std::to_string(i) + ".o";
        std::system( clean.c_str() );
      }

      std::string clean = " rm -R /tmp/nt2_external_kernel";
      // std::system(  (clean).c_str() );
      clean = "rm stderr.txt stdout.txt " + fname +".o";
      // std::system(  (clean).c_str() );


  return 0;
}
