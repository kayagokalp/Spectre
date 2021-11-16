#pragma once
#include "helpers.h"


class Material
{
	public:
		enum class CNT_TYPE {UD, FGV, FGO, FGX};
		enum class POROUS_TYPE {AA, BB, CC};

		Material(): mod_elasticity(0), poisson_ratio(0), mod_sheer_elasticity(0), 
			    elasticity{0,0,0},sheer_elasticity{0,0,0}, poissons{0,0,0},
		            CNT_type(CNT_TYPE::UD), POROUS_type(POROUS_TYPE::AA) {}
		Material(dtype _mod_elasticity,
				dtype _poisson_ratio,
				dtype _mod_sheer_elasticity, 
				dtype _elasticity_0,dtype _elasticity_1,dtype _elasticity_2,
			       	dtype _sh_elasticity_0, dtype _sh_elasticity_1, dtype _sh_elasticity_2,
			       	dtype _poisson_0, dtype _poisson_1, dtype _poisson_2,
				CNT_TYPE cnt_type,
				POROUS_TYPE porous_type,
				dtype _v_str_cnt = 0.11,
				dtype _ro_cnt = 1400,
				dtype _ro_m = 1150,
				dtype _eta_star_1 = 0.149,
				dtype _eta_star_2 = 0.934,
				dtype _eta_star_3 = 0.934,
				dtype _alpha = 0.6733,
				dtype _e = 0.1
			)
			: mod_elasticity(_mod_elasticity),
			poisson_ratio(_poisson_ratio),
			mod_sheer_elasticity(_mod_sheer_elasticity), 
			elasticity{_elasticity_0, _elasticity_1, _elasticity_2}, 
			sheer_elasticity{_sh_elasticity_0, _sh_elasticity_1, _sh_elasticity_2}, 
			poissons{_poisson_0, _poisson_1, _poisson_2},
			CNT_type(cnt_type), 
			POROUS_type(porous_type),
		        v_str_cnt(_v_str_cnt),
			ro_cnt(_ro_cnt),
			mod_ro(_ro_m),
			eta_star_1(_eta_star_1),
			eta_star_2(_eta_star_2),
			eta_star_3(_eta_star_3),
			alpha(_alpha),
			e(_e)	{}

		void operator=(const Material& s);
		
		//member variables
		dtype mod_elasticity; //E_m
		dtype elasticity[3]; // E_11 || E_22 || E_33 
		dtype poisson_ratio; //nu_m
		dtype poissons[3]; // nu12 || nu13 || nu23
		dtype mod_sheer_elasticity; // G_m
		dtype sheer_elasticity[3]; // G_11 || G_22 || G_33
		CNT_TYPE CNT_type; //TODO(kaya) : Convert this into enum
		POROUS_TYPE POROUS_type; //TODO(kaya) : Convert this into enum
		dtype v_str_cnt;
		dtype ro_cnt;
		dtype mod_ro;
		dtype eta_star_1;
		dtype eta_star_2;
		dtype eta_star_3;
		dtype alpha;
		dtype e;
};

