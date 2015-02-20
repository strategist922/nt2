#include <nt2/table.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/rand.hpp>
#include <nt2/include/functions/cons.hpp>
#include <nt2/include/functions/linsolve.hpp>

int main()
{
  using T = double;
  const std::size_t size_ =4;

  nt2::table<T> X = nt2::ones(nt2::of_size(size_), nt2::meta::as_<T>() );
  nt2::table<T,nt2::device_ > Y = X;

  NT2_DISPLAY(X);

  cublasDscal( size_, 5 , Y.data(), 1);

  X = Y;

  NT2_DISPLAY(X);

}

