#include "Material.h"



void Material::operator=(const Material& s)
{
  mod_elasticity = s.mod_elasticity;
  poisson_ratio = s.poisson_ratio;
  mod_sheer_elasticity = s.mod_sheer_elasticity;
  for(int i = 0; i<3; i++){
    elasticity[i] = s.elasticity[i];
    sheer_elasticity[i] = s.sheer_elasticity[i];
    poissons[i] = s.poissons[i];
  }
  CNT_type = s.CNT_type;
  POROUS_type = s.POROUS_type;
  v_str_cnt = s.v_str_cnt;
  ro_cnt = s.ro_cnt;
  mod_ro = s.mod_ro;
  eta_star_1 = s.eta_star_1;
  eta_star_2 = s.eta_star_2;
  eta_star_3 = s.eta_star_3;
  alpha = s.alpha;
  e = s.e;
}
