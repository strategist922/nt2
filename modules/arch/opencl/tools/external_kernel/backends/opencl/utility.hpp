#define rank "index"

// TODO : remove when c++11
std::string to_string (int i)
{
  std::ostringstream oss;
  oss << i;
  return oss.str ();
}

std::string erase_from (std::string const& pattern, std::string const& source)
{
  std::string result = source;
  int pos = result.find (pattern);
  if (pos != std::string::npos)
    result.erase (pos, pattern.size ());
  return result;
}

std::string erase_const_from (std::string const& source)
{
  return erase_from (" const", source);
}

std::string get_operation_params (expression const& expr)
{
  std::string op = expr.operation;
  size_t pos = op.find_last_of (":");
  op = op.substr (pos+1);
  if (expr.children.size () > 0)
    op += "(" + erase_const_from (expr.children[0].type.value_type);
      for (unsigned int i = 1; i < expr.children.size (); ++i)
      {
        op = op + "," + erase_const_from (expr.children[i].type.value_type);
      }
      if (expr.children.size () > 0)
        op += ")";
return op;
}

std::string get_value_type (expression const& expr)
{
  std::string value_type = erase_const_from(expr.type.value_type);

  if(expr.children.size() > 0)
  {
      value_type = erase_const_from (expr.children[0].type.value_type);
  }

return value_type;
}

template<class expr>
void display_info(expr const& e)
{
  std::cout <<"--------------- settings -----------------" << std::endl;
  std::cout << e.type << std::endl;
  std::cout << "operation : " << e.operation << std::endl;
  std::cout << "computed value_type : " << get_value_type(e) << std::endl;
  std::cout << "typed_operation : "  << get_operation_params(e) << std::endl;
  std::cout << "number of params : "  << e.children.size() << std::endl;
  for(std::size_t i = 0 ; i < e.children.size() ; ++i)
  {
    std::cout << "child " << i << " params : "
                              << e.children[i].children.size() << std::endl;
  }
}

bool is_terminal(std::string const& s) {return s =="terminal";}

// TODO optimize get_rhs
bool is_value(std::string const& s)
{
  return s =="double" || s =="float" ;
}

template<class Expr>
void is_on_device(Expr const& e, std::vector<std::size_t> & locality)
{
  std::string key = "nt2::device_";
  if(e.type.settings == key)
    locality.push_back(1);
  else locality.push_back(0);
}

template<class Expr>
void get_rhs_ops(int & t_ind, Expr const& rhs
            , std::vector<std::vector<std::string> >& accumulator
            )
{
  boost::format exp = boost::format("");
  exp = boost::format("%1% %2%(") % rhs.type.value_type % rhs.operation;
  std::vector<std::string> tmpAccu;
  tmpAccu.push_back(std::string(""));
  tmpAccu.push_back(rhs.operation);
  tmpAccu.push_back(rhs.type.value_type);

  for (std::size_t i = 0 ; i < rhs.children.size() ; ++i )
  {
    exp = boost::format("%1% %2% arg%3%,") % exp % rhs.children[i].type.value_type % to_string(i);
    std::string indx = "t" + to_string(t_ind);
    tmpAccu.push_back(rhs.children[i].type.value_type);

    if( !( is_terminal(rhs.children[i].operation) ) )
      get_rhs_ops(t_ind, rhs.children[i], accumulator);
  }
  std::string test = exp.str();
  test.pop_back();
  test = test + " )";

//  tmpAccu.push_back(test);
  tmpAccu[0] = test;
  accumulator.push_back(tmpAccu);
}

template<class Expr,class Format>
void get_rhs( int & t_ind, Expr const& rhs,Format & rhs_expr, Format & params
            , std::string & params_call , std::vector<std::size_t> & locality)
{
  for (std::size_t i = 0 ; i < rhs.children.size() ; ++i )
  {
    std::string indx = "t" + to_string(t_ind);

    if(is_terminal(rhs.children[i].operation))
    {
      t_ind++;
      if (rhs.children[i].type.container)
      {
        is_on_device(rhs.children[i], locality);
        rhs_expr = boost::format ("%1%%2%[%3%]") % rhs_expr % indx % rank;
        params = boost::format ("%1%, const %2%* __restrict %3%") % params % rhs.type.value_type % indx ;
        if(i != (rhs.children.size()-1) )
         rhs_expr = boost::format("%1%%2%") % rhs_expr % ",";
      }
      else if(is_value(rhs.children[i].type.value_type))// parameter is a constant
      {
        locality.push_back(2);
        rhs_expr = boost::format ("%1%%2%") % rhs_expr % indx ;
        params = boost::format ("%1%, const %2% %3%") % params % rhs.type.value_type % indx ;
        if(i != (rhs.children.size()-1) )
         rhs_expr = boost::format("%1%%2%") % rhs_expr % ",";

        // params_call += ",boost::proto::child_c<"+to_string(i)+">(a1)";
      }
      else
      {
        //TODO is_value is not generic but forced because of functions like nt2::ones
        //     that has multiple children with only the field value to separate

        t_ind--;
      }

    }
    else // add the function and call get_rhs recursively
    {
      rhs_expr = boost::format("%1%%2%(") % rhs_expr % rhs.children[i].operation ;
        get_rhs(t_ind, rhs.children[i], rhs_expr, params, params_call , locality);

      if( rhs.children.size() > i+1 )
        rhs_expr = boost::format("%1%),") % rhs_expr;
      else
        rhs_expr = boost::format("%1%)") % rhs_expr;
    }
  }
}


template<class Expr, class Format>
void get_lhs( int t_ind, Expr const& lhs , Format & e ,Format & params
            , std::string const& value_type , std::vector<std::size_t> & locality)
{
  std::string indx = "t" + to_string(t_ind);

  if(t_ind > 0) params = boost::format("%1% ,") % params ;

  if (lhs.type.container)
  {
    is_on_device(lhs,locality);
    e = boost::format ("%1%[%2%]") % indx % rank;
    params = boost::format ("%1% %2%* %3%") % params % value_type % indx ;
  }
  else
  {
    locality.push_back(1);
    e = boost::format ("%1%") % indx;
    params = boost::format ("%1% %2% %3%") % params % value_type % indx ;
  }
}

template<class Expr , class Map>
void add_map_includes(Expr const& rhs , Map & m, std::string const& s)
{
  if( !is_terminal(rhs.operation))
  {
    std::string temp = "#include <nt2/arithmetic/functions/" + s + "/"
                     + rhs.operation + ".hpp>\n" ;

    m.insert( std::pair<std::string,std::string>(rhs.operation, temp) );
  }

  for(std::size_t i = 0 ; i < rhs.children.size() ; ++i )
    add_map_includes(rhs.children[i], m, s);

}


std::set<std::string> opencl_fun_headers()
{
  std::set<std::string> s;

// TODO: add boost::compute headers
// Currently only has opnecl built-ins
  s.insert("acos");
  s.insert("acosh");
  s.insert("acospi");
  s.insert("asin");
  s.insert("asinh");
  s.insert("asinpi");
  s.insert("atan");
  s.insert("atan2");
  s.insert("atanh");
  s.insert("atanpi");
  s.insert("atan2pi");
  s.insert("cbrt");
  s.insert("ceil");
  s.insert("copysign");
  s.insert("cos");
  s.insert("cosh");
  s.insert("cospi");
  s.insert("erfc");
  s.insert("erf");
  s.insert("exp");
  s.insert("exp2");
  s.insert("exp10");
  s.insert("expm1");
  s.insert("fabs");
  s.insert("fdim");
  s.insert("floor");
  s.insert("fma");
  s.insert("fmax");
  s.insert("fmin");
  s.insert("fmod");
  s.insert("fract");
  s.insert("frexp");
  s.insert("hypot");
  s.insert("ilogb");
  s.insert("ldexp");
  s.insert("lgamma");
  s.insert("lgamma_r");
  s.insert("log");
  s.insert("log2");
  s.insert("log10");
  s.insert("log1p");
  s.insert("logb");
  s.insert("mad");
  s.insert("modf");
  s.insert("nan");
  s.insert("nextafter");
  s.insert("pow");
  s.insert("pown");
  s.insert("powr");
  s.insert("remainder");
  s.insert("remquo");
  s.insert("rint");
  s.insert("rootn");
  s.insert("round");
  s.insert("rsqrt");
  s.insert("sin");
  s.insert("sincos");
  s.insert("sinh");
  s.insert("sinpi");
  s.insert("sqrt");
  s.insert("tan");
  s.insert("tanh");
  s.insert("tanpi");
  s.insert("tgamma");

  return s;
}
