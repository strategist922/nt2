#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <boost/algorithm/string/replace.hpp>

void generate_resource(std::string const& symbol, const char* resource_name, std::istream& in, std::ostream& out)
{
  out << "#include <nt2/sdk/resource.hpp>\n\n";
  out << "char const " << symbol << "[] =\n{";

  std::istreambuf_iterator<char> it(in.rdbuf());
  std::istreambuf_iterator<char> end;
  std::size_t nb = 0;
  out << std::hex << std::setw(2) << std::setfill('0');
  for(; it != end; ++it, ++nb)
  {
    if(nb)
      out << ", ";
    if(nb % 10 == 0)
      out << "\n    ";
    out << std::setw(3) << std::setfill(' ') << std::dec << (int)*it;
  }

  out << "\n};\n\n";

  std::string resource = resource_name;
  boost::replace_all(resource, "\\", "\\\\");
  boost::replace_all(resource, "\"", "\\\"");
  out << "NT2_RESOURCE_REGISTER(\"" << resource << "\", " << symbol << ");\n";
}

int main(int argc, char* argv[])
{
  if(argc != 3)
  {
    std::cerr << "usage: " << argv[0] << " <resource_name> <path>" << std::endl;
    return 1;
  }

  std::ifstream in(argv[2]);
  if(!in)
  {
    std::cerr << "error: file `" << argv[2] << "' could not be opened for reading" << std::endl;
    return 1;
  }

  try
  {
    std::string symbol = std::string("nt2_resource_") + argv[1];
    std::replace(symbol.begin(), symbol.end(), '/', '_');
    std::replace(symbol.begin(), symbol.end(), '\\', '_');
    std::replace(symbol.begin(), symbol.end(), '.', '_');
    std::replace(symbol.begin(), symbol.end(), '-', '_');

    std::string outfile = symbol + ".cpp";
    std::ofstream out(outfile.c_str());
    if(!out)
    {
      std::cerr << "error: file `" << outfile << "' could not be opened for writing" << std::endl;
      return 1;
    }

    generate_resource(symbol, argv[1], in, out);
    return 0;
  }
  catch(std::exception const& e)
  {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << "error: uncaught exception" << std::endl;
    return 1;
  }
}
