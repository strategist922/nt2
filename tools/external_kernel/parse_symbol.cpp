#include "parse_symbol.hpp"

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <string>
#include "types.hpp"

namespace spirit  = boost::spirit;
namespace qi      = spirit::qi;
namespace phoenix = boost::phoenix;

typedef const char* char_iterator;
using boost::spirit::ascii::string;

struct grammar : qi::grammar<char_iterator, qi::space_type, kernel_symbol()>
{
  grammar(bool using_cl, bool debug) : base_type(root)
  {
    using namespace qi::labels;

    if(!using_cl)
    {
      header
        = qi::no_skip[*qi::space >> 'U' >> qi::space]
      ;
    }
    else
    {
      header
        = +qi::char_("0-9A-F")
          >> "UNDEF" >> "notype" >> "()" >> "External" >> "|"
          >> +qi::char_("a-zA-Z0-9?@$_") >> "(public: static "
      ;
    }

    root
      = ( header
          >> "void" >> -qi::lit("__cdecl") >> "nt2" >> "::" >> "external_kernel"
          >> '<'
            >> tag_name >> ','
            >>  tag_name >> (*( '<' >> +~qi::char_('>') >> '>' ) )
           >> qi::char_('>')
          >> "::" >> "call"
          >> '<'
            >> (tagged_expr % ',')
          >> '>'
        )
        [ _val = phoenix::construct<kernel_symbol>(_1, _2, _3, _5) ]
      ;

    tagged_expr = qi::raw[ expr [ _a = _1 ] ] [ _val = phoenix::construct<tagged_expression>(_1, _a) ];


    name_underscore = + ( ( qi::char_ - qi::char_("_,<>() \t\n") )
                        | ( qi::char_('_') >> &name_underscore  )
                        )
                    ;

    struct_class = qi::lit("struct") | qi::lit("class") | qi::eps;

    tag_name  =  struct_class
              >>  (   ( qi::lit("boost") >> "::" >> "simd"  )
                  |   ( qi::lit("boost") >> "::" >> "proto" >> "::" >> "tagns_" )
                  |   qi::lit("nt2")
                  )
              >> "::" >> "tag" >> "::" >> qi::lexeme[name_underscore] >> -qi::lit("_")
              ;

    name = +(qi::char_ - qi::char_(",<>()\t\n"));

    type_name
      =   -qi::lit("const")
     >>   struct_class
      >>  qi::raw[
            qi::lexeme[name]
            >> -( qi::lit('<')
                  >> type_name % ','
                  >> '>'
                )
            >> -( -qi::lit("__cdecl") >> qi::lit('(')
                  >> (qi::lit("void") | type_name % ',' | qi::eps)
                  >> ')'
                )
          ]
          >> -qi::lit("const")
          >> -qi::lit('&')
      ;

    proto_expr
      = ( struct_class >> qi::lit("nt2") >> "::" >> "container" >> "::" >> "expression"
          >> '<'
            >> struct_class >> "boost" >> "::" >> "proto" >> "::" >> "exprns_" >> "::" >> "basic_expr"
            >> '<'
              >> tag_name >> ','
              >> proto_expr_children >> ','
              >> qi::omit[qi::int_ >> -qi::char_("lLuU")]
            >> '>' >> ','
            >> type
          >> '>'
        ) [ _val = phoenix::construct<expression>(_1, _3, _2) ]
      ;

    proto_expr_children
      = struct_class >> qi::lit("boost") >> "::" >> "proto" >> "::" >> "argsns_" >> "::"
        >> (  ( qi::lit("term") >> '<' >> type_name >> '>' ) [ _val = phoenix::construct< std::vector<expression> >() ]
           |  ( qi::lit("list") >> qi::omit[qi::int_]
                >> '<'
                  >> (expr % ',')
                >> '>'
              ) [ _val = _1 ]
           )
      ;

    expr =  (  proto_expr [ _val = _1 ]
            |  ( ( struct_class >> qi::lit("nt2") >> "::" >> "container" >> "::" )
                  >> qi::lit("table")
                  >> type_container
               ) [ _val = phoenix::construct<expression>(std::string("terminal"), _1, std::vector<expression>()) ]
            |  ( ( struct_class >> qi::lit("nt2") >> "::" >> "container" >> "::" )
                  >>  ( qi::lit("view")
                      | qi::lit("shared_view")
                      )
                  >> qi::lit("<") >> qi::lit("nt2") >> "::" >> "container" >> "::" >> "table"
                  >> type_container >> -qi::lit("const") >> ">"
               ) [ _val = phoenix::construct<expression>(std::string("terminal"), _1, std::vector<expression>()) ]
            |  ( struct_class >> qi::lit("nt2") >> "::" >> "box" >> '<'
                  >> type_name
                  >> '>'
               ) [ _val = phoenix::construct<expression>(std::string("terminal"), _1, std::vector<expression>()) ]
            |  (+qi::char_("a-zA-Z:") >> "<" >> (+qi::char_("a-zA-Z")) >> ">") [_val = phoenix::construct<expression>(std::string("type"), _1, _2, std::vector<expression>()) ]
            |  tag_name [  _val = phoenix::construct<expression>(std::string("tag"), _1, std::vector<expression>()) ]
            |  name [_val = phoenix::construct<expression>(std::string("type"), _1, std::vector<expression>()) ]
            )
            >> -qi::lit("const")
         ;

    type
      = ( -qi::lit("const") >> struct_class >> qi::lit("nt2") >> "::" >> "memory" >> "::" >> "container" >> type_container >> -qi::lit("const") >> -qi::lit('&') ) [ _val = _1 ]
         | type_name [ _val = phoenix::construct<type_infos>(_1) ]
      ;

    type_container
      = ( qi::lit('<')
          >> -(qi::lit("nt2::tag::table_") >> ',')
          >> type_name >> ','
          >> type_name
          >> qi::lit('>')
        ) [ _val = phoenix::construct<type_infos>(_1, true,_2) ]
      ;

    #define NT2_SPIRIT_DEBUG_NODE(r)                                            \
    r.name(#r);                                                                 \
    if(debug)                                                                   \
      boost::spirit::qi::debug(r);                                              \
    /**/

    NT2_SPIRIT_DEBUG_NODE(header);
    NT2_SPIRIT_DEBUG_NODE(struct_class);
    NT2_SPIRIT_DEBUG_NODE(root);
    NT2_SPIRIT_DEBUG_NODE(tagged_expr);
    NT2_SPIRIT_DEBUG_NODE(name);
    NT2_SPIRIT_DEBUG_NODE(tag_name);
    NT2_SPIRIT_DEBUG_NODE(name_underscore);
    NT2_SPIRIT_DEBUG_NODE(type_name);
    NT2_SPIRIT_DEBUG_NODE(expr);
    NT2_SPIRIT_DEBUG_NODE(proto_expr);
    NT2_SPIRIT_DEBUG_NODE(proto_expr_children);
    NT2_SPIRIT_DEBUG_NODE(type);
    NT2_SPIRIT_DEBUG_NODE(type_container);
  }

  std::vector<std::string> test;
  qi::rule<char_iterator, qi::space_type> header;
  qi::rule<char_iterator, qi::space_type> struct_class;
  qi::rule<char_iterator, qi::space_type, kernel_symbol()           > root;
  qi::rule<char_iterator, qi::space_type, tagged_expression(), qi::locals<expression> > tagged_expr;
  qi::rule<char_iterator, std::string()                             > name;
  qi::rule<char_iterator, std::string()                             > name_underscore;
  qi::rule<char_iterator, qi::space_type, std::string()             > tag_name;
  qi::rule<char_iterator, qi::space_type, std::string()             > type_name;
  qi::rule<char_iterator, qi::space_type, expression()              > expr;
  qi::rule<char_iterator, qi::space_type, expression()              > proto_expr;
  qi::rule<char_iterator, qi::space_type, std::vector<expression>() > proto_expr_children;
  qi::rule<char_iterator, qi::space_type, type_infos()              > type;
  qi::rule<char_iterator, qi::space_type, type_infos()              > type_container;
};

bool parse_symbol(bool using_cl, bool debug, std::string const& line, kernel_symbol& symbol)
{
  const char* begin = line.c_str();
  return qi::phrase_parse(begin, begin+line.size(), grammar(using_cl, debug), qi::space, symbol);
}
