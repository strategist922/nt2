#include <nt2/include/functions/transpose.hpp>
#include <nt2/include/functions/tksolve.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/eye.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/cons.hpp>

#include <nt2/table.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>

NT2_TEST_CASE_TPL(tksolve, NT2_REAL_TYPES )
{
  using nt2::_;

  typedef nt2::table<T>         t_t;

  t_t a = nt2::eye(3, 3, nt2::meta::as_<T>());
  t_t b = nt2::ones(9, 1,  nt2::meta::as_<T>());
  t_t r = b*nt2::Half<T>();
  t_t rr;

  rr = tksolve(a, b, 'N');
  NT2_TEST_ULP_EQUAL(rr, r, 0.5);
  rr = tksolve(a, b, 'T');
  NT2_TEST_ULP_EQUAL(rr , r, 0.5);

  a = nt2::eye(3, 3, nt2::meta::as_<T>());
  a(2, 2) = T(2);
  a(3, 3) = T(3);
  b = nt2::ones(9, 1,  nt2::meta::as_<T>());
  b(_(1, 2, 9)) = T(2)* b(_(1, 2, 9));
  r = nt2::cons < T > (
    1.0,
    1.0/T(3),
    0.5000,
    1.0/T(3),
    0.5000,
    0.2000,
    0.5000,
    0.2000,
    1.0/T(3));
  rr = tksolve(a, b, 'N');
  NT2_TEST_ULP_EQUAL(rr, r, 0.5);
  rr = tksolve(a, b, 'T');
  NT2_TEST_ULP_EQUAL(rr, r, 0.5);

  a = nt2::eye(3, 3, nt2::meta::as_<T>());
  a(1, 3) = T(0.5);
  r = nt2::cons < T > (
    0.7500,
    0.5000,
    1.0000,
    0.3750,
    1.0000,
    0.5000,
    0.6250,
    0.3750,
    0.7500 );
  rr = tksolve(a, b, 'N');
  NT2_TEST_ULP_EQUAL(rr, r, 0.5);
  r = nt2::cons < T > (
    0.7500,
    0.3750,
    0.6250,
    0.5000,
    1.0000,
    0.3750,
    1.0000,
    0.5000,
    0.7500 );
  rr = tksolve(a, b, 'T');
  NT2_TEST_ULP_EQUAL(rr, r, 0.5);
}
