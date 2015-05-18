#ifndef PARSE_SYMBOL_HPP_INCLUDED
#define PARSE_SYMBOL_HPP_INCLUDED

#include <string>
#include "types.hpp"

bool parse_symbol(bool using_cl, bool debug, std::string const& line, kernel_symbol& symbol);

#endif
