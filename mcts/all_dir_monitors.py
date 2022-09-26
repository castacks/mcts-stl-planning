import torch
import numpy as np
# import signal_tl as stl

import sys
import rtamt


def goal_to_spec(goal):

	stl_func = "monitor_" + goal[0]
	print("STL Func: " , stl_func)
	STL = globals()[stl_func]

	return STL

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



def monitor_R1(ways): # rtamt specs for runway R2
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


	spec.spec = 'eventually( (( (x >-1.5) and (x < 1.5)) and ( (y < 3) and (y > 2) )   and (z<0.5) ) and (eventually( (((x < -4.5) and (x > -5.0) ) and ((y > -3) and (y < 0.2) and (z<0.5))  ) and (eventually( ((x < 0.2) and (x > -0.2) ) and ((y > -0.2) and (y < 0.2) and z<0.5 ) )))))'



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



def monitor_N(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	# region_N_upper = np.array([[1.48,0.0,0.256]])
	# region_N_upper = np.array([[1.48,0.0,0.256]])
	region_N = np.array([[0.0,10.0,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_N.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_N, 2), 1).reshape(region_N.shape[0], 1), (region_N.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_N = np.sqrt(sqrA - 2*np.matmul(ways, region_N.transpose()) + sqrB)
	#
	# print(L2_dist_N.shape)
	L2_dist_N = L2_dist_N.squeeze().tolist()

	L2_north = list(enumerate(L2_dist_N))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'north goal'

	spec.declare_var('north', 'float')

	spec.spec = 'eventually(north < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['north',L2_north])
	return out[0][1]


def monitor_S(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_S = np.array([[0.0,-10.0,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_S.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_S, 2), 1).reshape(region_S.shape[0], 1), (region_S.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_S = np.sqrt(sqrA - 2*np.matmul(ways, region_S.transpose()) + sqrB)
	#
	L2_dist_S = L2_dist_S.squeeze().tolist()

	L2_south = list(enumerate(L2_dist_S))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'south goal'

	spec.declare_var('south', 'float')

	spec.spec = 'eventually(south < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['south',L2_south])
	return out[0][1]


def monitor_E(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_E = np.array([[10.0,0.0,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_E.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_E, 2), 1).reshape(region_E.shape[0], 1), (region_E.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_E = np.sqrt(sqrA - 2*np.matmul(ways, region_E.transpose()) + sqrB)
	#
	L2_dist_E = L2_dist_E.squeeze().tolist()

	L2_east = list(enumerate(L2_dist_E))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'east goal'

	spec.declare_var('east', 'float')

	spec.spec = 'eventually(east < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['east',L2_east])
	return out[0][1]


def monitor_W(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_W = np.array([[-10.0,0.0,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_W.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_W, 2), 1).reshape(region_W.shape[0], 1), (region_W.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_W = np.sqrt(sqrA - 2*np.matmul(ways, region_W.transpose()) + sqrB)
	#
	L2_dist_W = L2_dist_W.squeeze().tolist()

	L2_west = list(enumerate(L2_dist_W))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'west goal'

	spec.declare_var('west', 'float')

	spec.spec = 'eventually(west < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['west',L2_west])
	return out[0][1]


def monitor_NE(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	# region_N_upper = np.array([[1.48,0.0,0.256]])
	# region_N_upper = np.array([[1.48,0.0,0.256]])
	region_NE = np.array([[7.071,7.071,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_NE.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_NE, 2), 1).reshape(region_NE.shape[0], 1), (region_NE.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_NE = np.sqrt(sqrA - 2*np.matmul(ways, region_NE.transpose()) + sqrB)
	#
	# print(L2_dist_N.shape)
	L2_dist_NE = L2_dist_NE.squeeze().tolist()

	L2_northeast = list(enumerate(L2_dist_NE))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'northeast goal'

	spec.declare_var('northeast', 'float')

	spec.spec = 'eventually(northeast < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['northeast',L2_northeast])
	return out[0][1]


def monitor_SE(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_SE = np.array([[7.071,-7.071,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_SE.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_SE, 2), 1).reshape(region_SE.shape[0], 1), (region_SE.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_SE = np.sqrt(sqrA - 2*np.matmul(ways, region_SE.transpose()) + sqrB)
	#
	L2_dist_SE = L2_dist_SE.squeeze().tolist()

	L2_southeast = list(enumerate(L2_dist_SE))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'southeast goal'

	spec.declare_var('southeast', 'float')

	spec.spec = 'eventually(southeast < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['southeast',L2_southeast])
	return out[0][1]


def monitor_NW(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_NW = np.array([[-7.071,7.071,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_NW.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_NW, 2), 1).reshape(region_NW.shape[0], 1), (region_NW.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_NW = np.sqrt(sqrA - 2*np.matmul(ways, region_NW.transpose()) + sqrB)
	#
	L2_dist_NW = L2_dist_NW.squeeze().tolist()

	L2_northwest = list(enumerate(L2_dist_NW))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'northwest goal'

	spec.declare_var('northwest', 'float')

	spec.spec = 'eventually(northwest < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['northwest',L2_northwest])
	return out[0][1]


def monitor_SW(ways):
	trajs_x = ways[:,0].tolist()
	trajs_y = (ways[:,1]).tolist()
	trajs_z = (ways[:,2]).tolist()

	region_SW = np.array([[-7.071,-7.071,2.5]])


	if torch.is_tensor(ways):
		ways = ways.numpy()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], region_SW.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(region_SW, 2), 1).reshape(region_SW.shape[0], 1), (region_SW.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_SW = np.sqrt(sqrA - 2*np.matmul(ways, region_SW.transpose()) + sqrB)
	#
	L2_dist_SW = L2_dist_SW.squeeze().tolist()

	L2_southwest = list(enumerate(L2_dist_SW))

	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	spec.name = 'southwest goal'

	spec.declare_var('southwest', 'float')

	spec.spec = 'eventually(southwest < 0.2) '


	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	out = spec.evaluate(['southwest',L2_southwest])
	return out[0][1]