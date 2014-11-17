//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/core/settings/add_settings.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>

NT2_TEST_CASE( add_nothing_to_settings )
{
  using nt2::settings;
  using nt2::meta::add_settings;

  using empty = settings();
  using base  = settings(int);

  NT2_TEST_TYPE_IS( add_settings<empty>::type         , empty );
  NT2_TEST_TYPE_IS( add_settings<base>::type          , int   );
  NT2_TEST_TYPE_IS( (add_settings<empty,empty>::type) , empty );
  NT2_TEST_TYPE_IS( (add_settings<base,empty>::type)  , int   );
  NT2_TEST_TYPE_IS( (add_settings<empty,base>::type)  , int   );
}

NT2_TEST_CASE( add_settings_to_nothing )
{
  using nt2::settings;
  using nt2::meta::add_settings;

  using empty = settings();

  NT2_TEST_TYPE_IS( (add_settings<empty, int>::type), int );
}

NT2_TEST_CASE( add_option_to_setting )
{
  using nt2::settings;
  using nt2::meta::add_settings;
  using base = settings(float);
  NT2_TEST_TYPE_IS( (add_settings<float, int>::type), (settings(int,float)) );
  NT2_TEST_TYPE_IS( (add_settings<base, int>::type) , (settings(int,float)) );
}

NT2_TEST_CASE( add_options_to_setting )
{
  using nt2::settings;
  using nt2::meta::add_settings;
  using base  = settings(float);
  using newer = settings(int);

  NT2_TEST_TYPE_IS( (add_settings<base, newer>::type), (settings(int,float)) );
}

NT2_TEST_CASE( add_options_to_settings )
{
  using nt2::settings;
  using nt2::meta::add_settings;

  using base    = settings(void*);
  using base2   = settings(void*,void**);
  using new2    = settings(int,float);
  using new3    = settings(int,float,char);
  using new4    = settings(int,float,char,double);

  NT2_TEST_TYPE_IS( (add_settings<base, new2>::type)
                  , (settings(int,float,void*))
                  );

  NT2_TEST_TYPE_IS( (add_settings<base, new3>::type)
                  , (settings(int,float,char,void*))
                  );

  NT2_TEST_TYPE_IS( (add_settings<base, new4>::type)
                  , (settings(int,float,char,double,void*))
                  );

  NT2_TEST_TYPE_IS( (add_settings<base2, new2>::type)
                  , (settings(int,float,void*,void**))
                  );

  NT2_TEST_TYPE_IS( (add_settings<base2, new3>::type)
                  , (settings(int,float,char,void*,void**))
                  );

  NT2_TEST_TYPE_IS( (add_settings<base2, new4>::type)
                  , (settings(int,float,char,double,void*,void**))
                  );
}
