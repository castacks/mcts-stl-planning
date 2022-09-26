import torch
import numpy as np
# import signal_tl as stl

import sys
import rtamt
"""
Runway R2 ( not fully updated)
Spec 1: phi_1 = downwind: x [0.5, 3.0]
						  y [-2.5, -0.8]
						  z [0.5, 0.7]

Spec 2: phi_2 = base:     x [2, 4.0]
						  y [-1.8, 0.02]
						  z [0.3, 0.5]

Spec 3: phi_3 = final     x [1.7, 3.7]
						  y [-0.08, 0.08]
						  x [0.3, 0.5]

Spec 4: psi = overflying  x [0.0, 1.45]
						  y [-0.01, 0.01]
						  z [0.1, 0.6]

"""



def monitor_R2(ways): # rtamt specs for runway R2
	# data
	# print("Hi", trajs0.shape)
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]*1.1).tolist()
	trajs_z = (ways[:,2]*1.2).tolist()

	# user_details = [{'name' : x, 'rank' : trajs_x.index(x)} for x in trajs_x]
	trajs_x = list(enumerate(trajs_x))
	trajs_y = list(enumerate(trajs_y))
	trajs_z = list(enumerate(trajs_z))



	# # stl
	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'runway r2'

	spec.declare_var('x', 'float')
	spec.declare_var('y', 'float')
	spec.declare_var('z', 'float')


	spec.spec = 'eventually( (( (x >-1.5) and (x < 1.5)) and ( (y > -3) and (y < -2) )   and (z<0.5) ) and (eventually( (((x > 4.5) and (x < 5.0) ) and ((y > -3) and (y < 0.2) and (z<0.5))  ) and (eventually( ((x > 1.3) and (x < 1.5) ) and ((y > -0.2) and (y < 0.2) and z<0.5 ) )))))'



	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['x',trajs_x],['y',trajs_y],['z',trajs_z])

	# out = spec.evaluate(['x', trajs_x],['y', trajs_y],['diff_tan',diff_tan] ,['phi_l',phi_l],['phi_u',phi_u],['phi_l2',phi_l2],['phi_u2',phi_u2],['phi_l3',phi_l3],['phi_u3',phi_u3],['vel_x',vel_x],['vel_y',vel_y],['vel_z',vel_z])
	# out = spec.evaluate(['x', trajs_x],['y', trajs_y],['z', trajs_z],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z])
	# print('Robustness offline: {}'.format(out))
	return out[0][1]


