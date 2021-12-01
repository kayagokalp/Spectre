#include "Space.h"



Space::Space(dtype start, dtype end, int no_points)
: start(start), end(end), no_points(no_points),
	IT(no_points, no_points), FT(no_points, no_points), D(no_points, no_points), s(no_points)
{
		IT.setZero();
		FT.setZero();
		D.setZero();
		s.setZero();
		V.setZero();
		Q1.setZero();
		discretize();
}


void Space::discretize()
{
  cheb(no_points, IT, FT);
  DBG(cout << "IT\n"
      << IT << endl;);
  DBG(cout << "FT\n"
      << FT << endl;);
  derivative(start, end, no_points, D);
  DBG(cout << "D\n"
      << D << endl;);
  slobat(start, end, no_points, s);
  DBG(cout << "s\n"
      << s << endl;);
  inner_product_helper(start, end, no_points, V);
  DBG(cout << "V\n"
      << V << endl;);
  Q1 = IT * D * FT;
  DBG(cout << "Q1\n"
      << Q1 << endl;);
}
