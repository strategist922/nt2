#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
  std::vector<std::string> paths;

  for(int i = 1; i != argc; ++i)
  {
      // include directories, two argument syntax
      if(!strcmp(argv[i], "--path") && i != argc-1)
      {
          paths.push_back(argv[i+1]);
          ++i;
          continue;
      }

      if(!strcmp(argv[i], "--build_dir"))
      {
          paths.push_back(argv[i+1]);
          ++i;
          continue;
      }

      if(!strcmp(argv[i], "--source_dir"))
      {
          paths.push_back(argv[i+1]);
          ++i;
          continue;
      }

  }
  system( (paths[0]+" "+paths[1]).c_str() );

  return 0;
}
