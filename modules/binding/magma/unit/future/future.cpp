#define NT2_UNIT_MODULE "nt2 future gpu"

#include <nt2/table.hpp>
#include <nt2/sdk/magma/future.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>
#include "Obj_cuda.hpp"

using Site = typename boost::dispatch::default_site<void>::type;
using Arch = typename nt2::tag::magma_<Site>;
using future_1 = typename nt2::make_future<Arch,int>::type;

NT2_TEST_CASE( future_get )
{
  future_1 f1 = nt2::async<Arch>(Obj_cuda(),200);
  int value = f1.get();

  NT2_TEST_EQUAL(value,230);
}
