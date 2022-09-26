import torch
import numpy as np
# import signal_tl as stl

import sys
import rtamt
import time
from joblib import Parallel, delayed




trajs10 = torch.tensor([[-1.5650,  0.1290,  0.6401],
	[-1.6037,  0.1343,  0.6401],
	[-1.6423,  0.1390,  0.6401],
	[-1.6807,  0.1431,  0.6401],
	[-1.7189,  0.1467,  0.6401],
	[-1.7569,  0.1497,  0.6401],
	[-1.7947,  0.1521,  0.6401],
	[-1.8323,  0.1539,  0.6401],
	[-1.8697,  0.1552,  0.6401],
	[-1.9069,  0.1559,  0.6401],
	[-1.9440,  0.1561,  0.6401],
	[-1.9808,  0.1556,  0.6401],
	[-2.0175,  0.1546,  0.6401],
	[-2.0539,  0.1531,  0.6401],
	[-2.0902,  0.1509,  0.6401],
	[-2.1262,  0.1482,  0.6401],
	[-2.1621,  0.1449,  0.6401],
	[-2.1978,  0.1411,  0.6401],
	[-2.2333,  0.1366,  0.6401],
	[-2.2686,  0.1316,  0.6401],
	[-2.2686e+00,  1.3163e-01,  6.4008e-01],
	[-2.3327e+00,  1.2163e-01,  6.4275e-01],
	[-2.3964e+00,  1.0875e-01,  6.4542e-01],
	[-2.4593e+00,  9.3034e-02,  6.4809e-01],
	[-2.5216e+00,  7.4509e-02,  6.5077e-01],
	[-2.5829e+00,  5.3213e-02,  6.5344e-01],
	[-2.6432e+00,  2.9188e-02,  6.5611e-01],
	[-2.7024e+00,  2.4827e-03,  6.5878e-01],
	[-2.7603e+00, -2.6850e-02,  6.6145e-01],
	[-2.8168e+00, -5.8750e-02,  6.6412e-01],
	[-2.8719e+00, -9.3154e-02,  6.6680e-01],
	[-2.9254e+00, -1.2999e-01,  6.6947e-01],
	[-2.9771e+00, -1.6919e-01,  6.7214e-01],
	[-3.0271e+00, -2.1067e-01,  6.7481e-01],
	[-3.0751e+00, -2.5435e-01,  6.7748e-01],
	[-3.1211e+00, -3.0014e-01,  6.8015e-01],
	[-3.1650e+00, -3.4794e-01,  6.8283e-01],
	[-3.2068e+00, -3.9767e-01,  6.8550e-01],
	[-3.2462e+00, -4.4922e-01,  6.8817e-01],
	[-3.2834e+00, -5.0248e-01,  6.9084e-01],
	[-3.2834, -0.5025,  0.6908],
	[-3.3077, -0.5383,  0.6908],
	[-3.3296, -0.5757,  0.6908],
	[-3.3490, -0.6144,  0.6908],
	[-3.3657, -0.6544,  0.6908],
	[-3.3796, -0.6954,  0.6908],
	[-3.3908, -0.7372,  0.6908],
	[-3.3992, -0.7797,  0.6908],
	[-3.4046, -0.8227,  0.6908],
	[-3.4072, -0.8659,  0.6908],
	[-3.4069, -0.9092,  0.6908],
	[-3.4036, -0.9524,  0.6908],
	[-3.3975, -0.9953,  0.6908],
	[-3.3885, -1.0377,  0.6908],
	[-3.3767, -1.0793,  0.6908],
	[-3.3621, -1.1201,  0.6908],
	[-3.3448, -1.1598,  0.6908],
	[-3.3248, -1.1983,  0.6908],
	[-3.3024, -1.2353,  0.6908],
	[-3.2774, -1.2707,  0.6908], 
	[-3.2774, -1.2707,  0.6908],
	[-3.2595, -1.2978,  0.6922],
	[-3.2413, -1.3247,  0.6935],
	[-3.2227, -1.3513,  0.6948],
	[-3.2039, -1.3777,  0.6962],
	[-3.1847, -1.4039,  0.6975],
	[-3.1652, -1.4299,  0.6989],
	[-3.1454, -1.4556,  0.7002],
	[-3.1253, -1.4811,  0.7015],
	[-3.1050, -1.5064,  0.7029],
	[-3.0843, -1.5314,  0.7042],
	[-3.0633, -1.5562,  0.7055],
	[-3.0421, -1.5808,  0.7069],
	[-3.0206, -1.6051,  0.7082],
	[-2.9987, -1.6291,  0.7095],
	[-2.9766, -1.6529,  0.7109],
	[-2.9543, -1.6764,  0.7122],
	[-2.9316, -1.6997,  0.7136],
	[-2.9087, -1.7227,  0.7149],
	[-2.8855, -1.7454,  0.7162], 
	[-2.8855, -1.7454,  0.7162],
	[-2.8624, -1.7680,  0.7189],
	[-2.8390, -1.7904,  0.7216],
	[-2.8153, -1.8126,  0.7242],
	[-2.7914, -1.8344,  0.7269],
	[-2.7673, -1.8559,  0.7296],
	[-2.7429, -1.8772,  0.7323],
	[-2.7182, -1.8982,  0.7349],
	[-2.6933, -1.9189,  0.7376],
	[-2.6681, -1.9393,  0.7403],
	[-2.6428, -1.9594,  0.7429],
	[-2.6171, -1.9792,  0.7456],
	[-2.5913, -1.9987,  0.7483],
	[-2.5652, -2.0179,  0.7510],
	[-2.5389, -2.0368,  0.7536],
	[-2.5124, -2.0554,  0.7563],
	[-2.4857, -2.0736,  0.7590],
	[-2.4587, -2.0916,  0.7616],
	[-2.4316, -2.1092,  0.7643],
	[-2.4042, -2.1266,  0.7670], 
	[-2.4042, -2.1266,  0.7670],
	[-2.3768, -2.1439,  0.7683],
	[-2.3491, -2.1610,  0.7697],
	[-2.3213, -2.1777,  0.7710],
	[-2.2933, -2.1941,  0.7723],
	[-2.2651, -2.2101,  0.7737],
	[-2.2367, -2.2259,  0.7750],
	[-2.2081, -2.2413,  0.7763],
	[-2.1793, -2.2563,  0.7777],
	[-2.1504, -2.2710,  0.7790],
	[-2.1213, -2.2854,  0.7803],
	[-2.0920, -2.2995,  0.7817],
	[-2.0626, -2.3132,  0.7830],
	[-2.0330, -2.3265,  0.7843],
	[-2.0033, -2.3395,  0.7857],
	[-1.9734, -2.3522,  0.7870],
	[-1.9433, -2.3645,  0.7884],
	[-1.9132, -2.3765,  0.7897],
	[-1.8828, -2.3881,  0.7910],
	[-1.8524, -2.3993,  0.7924], 
	[-1.8524, -2.3993,  0.7924],
	[-1.8219, -2.4106,  0.7937],
	[-1.7914, -2.4215,  0.7950],
	[-1.7606, -2.4320,  0.7964],
	[-1.7298, -2.4422,  0.7977],
	[-1.6989, -2.4520,  0.7990],
	[-1.6678, -2.4614,  0.8004],
	[-1.6366, -2.4705,  0.8017],
	[-1.6054, -2.4793,  0.8031],
	[-1.5740, -2.4876,  0.8044],
	[-1.5425, -2.4956,  0.8057],
	[-1.5110, -2.5032,  0.8071],
	[-1.4793, -2.5105,  0.8084],
	[-1.4476, -2.5173,  0.8097],
	[-1.4158, -2.5239,  0.8111],
	[-1.3839, -2.5300,  0.8124],
	[-1.3520, -2.5358,  0.8137],
	[-1.3200, -2.5411,  0.8151],
	[-1.2879, -2.5462,  0.8164],
	[-1.2558, -2.5508,  0.8177],
	[-1.2558, -2.5508,  0.8177],
	[-1.2236, -2.5554,  0.8191],
	[-1.1915, -2.5597,  0.8204],
	[-1.1592, -2.5636,  0.8218],
	[-1.1270, -2.5671,  0.8231],
	[-1.0946, -2.5702,  0.8244],
	[-1.0623, -2.5730,  0.8258],
	[-1.0299, -2.5754,  0.8271],
	[-0.9975, -2.5774,  0.8284],
	[-0.9651, -2.5790,  0.8298],
	[-0.9327, -2.5802,  0.8311],
	[-0.9002, -2.5811,  0.8324],
	[-0.8677, -2.5816,  0.8338],
	[-0.8353, -2.5817,  0.8351],
	[-0.8028, -2.5814,  0.8364],
	[-0.7704, -2.5807,  0.8378],
	[-0.7379, -2.5797,  0.8391],
	[-0.7055, -2.5783,  0.8405],
	[-0.6731, -2.5765,  0.8418],
	[-0.6407, -2.5743,  0.8431], 
	[-0.6407, -2.5743,  0.8431],
	[-0.6083, -2.5721,  0.8445],
	[-0.5759, -2.5696,  0.8458],
	[-0.5436, -2.5666,  0.8471],
	[-0.5113, -2.5633,  0.8485],
	[-0.4790, -2.5596,  0.8498],
	[-0.4468, -2.5556,  0.8511],
	[-0.4147, -2.5511,  0.8525],
	[-0.3826, -2.5463,  0.8538],
	[-0.3505, -2.5411,  0.8551],
	[-0.3185, -2.5355,  0.8565],
	[-0.2866, -2.5296,  0.8578],
	[-0.2548, -2.5233,  0.8592],
	[-0.2230, -2.5166,  0.8605],
	[-0.1913, -2.5096,  0.8618],
	[-0.1597, -2.5021,  0.8632],
	[-0.1282, -2.4943,  0.8645],
	[-0.0968, -2.4862,  0.8658],
	[-0.0655, -2.4776,  0.8672],
	[-0.0342, -2.4688,  0.8685],
	[-0.0342, -2.4688,  0.8685],
	[ 0.0071, -2.4557,  0.8685],
	[ 0.0474, -2.4400,  0.8685],
	[ 0.0866, -2.4215,  0.8685],
	[ 0.1245, -2.4005,  0.8685],
	[ 0.1608, -2.3770,  0.8685],
	[ 0.1955, -2.3511,  0.8685],
	[ 0.2284, -2.3229,  0.8685],
	[ 0.2593, -2.2925,  0.8685],
	[ 0.2881, -2.2602,  0.8685],
	[ 0.3147, -2.2260,  0.8685],
	[ 0.3389, -2.1901,  0.8685],
	[ 0.3607, -2.1526,  0.8685],
	[ 0.3799, -2.1138,  0.8685],
	[ 0.3964, -2.0737,  0.8685],
	[ 0.4102, -2.0327,  0.8685],
	[ 0.4212, -1.9908,  0.8685],
	[ 0.5294, -1.9483,  0.8685],
	[ 1.5347, -1.9053,  0.8685],
	[ 3.5371, -1.8620,  0.8685]])


phi_1_x = [0.5, 2.50]
phi_1_y = [-2.5, -1.0]
phi_1_z = [0.5, 0.7]
phi_2_x = [2.5, 4.0]
phi_2_y = [-2.5, -1.0] 
phi_2_z = [0.3, 0.5]
phi_3_x = [1.5, 4.0]# region 3 isolated for testing purposes
phi_3_y = [-0.5, 0.2]
phi_3_z = [0.3, 0.5]

psi_x = [0.0, 1.45]
psi_y = [-0.01, 0.01]
psi_z = [0.1, 0.6]


def monitor_R2(ways): # rtamt specs for runway R2
	# data
	# print("Hi", trajs0.shape)
	trajs_x = ways[:,0].tolist()
	trajs_y = ways[:,1].tolist()
	trajs_z = ways[:,2].tolist()
	r2 = np.array([[1.48,0.0,0.38]])
	diff_trajs = np.diff(ways, axis = 0)
	# print (diff_trajs,diff_trajs[:,1])
	diff_angle = np.arctan2(diff_trajs[:,1],diff_trajs[:,0])
	# print (diff_angle*180/np.pi)
	# print (diff_angle.shape)

	diff_angle_full = np.concatenate((diff_angle,diff_angle[-1:]))
	# print (diff_angle_full*180/np.pi)

	diff_angle_full = np.where(diff_angle_full < -1.57, diff_angle_full + 2 * np.pi, diff_angle_full)

	# diff_angle_full = diff_angle_full.tolist()
	phi_1u = 0.349066 - diff_angle_full
	phi_1l = diff_angle_full + 0.349066
	phi_2u = 2.00713 - diff_angle_full
	phi_2l = diff_angle_full - 1.13446
	phi_3u = 3.40339 - diff_angle_full
	phi_3l = diff_angle_full - 2.87979

	phi_1u = phi_1u.tolist()
	phi_1l = phi_1l.tolist()
	phi_2u = phi_2u.tolist()
	phi_2l = phi_2l.tolist()
	phi_3u = phi_3u.tolist()
	phi_3l = phi_3l.tolist()

	diff_trajs = np.concatenate((diff_trajs,diff_trajs[-1:]))
	# print (diff_angle_full*180/np.pi)
	# print('phi_3l',phi_3l)

	vels_x = diff_trajs[:,0].tolist()
	vels_y = diff_trajs[:,1].tolist()
	vels_z = diff_trajs[:,2].tolist()

	# print(len(vels_x))

	# diff_from_runway = ways - r2
	# print(diff_from_runway)
	# print("ways",ways.shape,r2.shape)
	if torch.is_tensor(ways):
		ways = ways.numpy()
	# sqrA = torch.sum(torch.pow(ways, 2), 1, keepdim=True).expand(ways.shape[0], r2.shape[0])
	# sqrB = torch.sum(torch.pow(r2, 2), 1, keepdim=True).expand(r2.shape[0], ways.shape[0]).t()

	sqrA = np.broadcast_to(np.sum(np.power(ways, 2), 1).reshape(ways.shape[0], 1), (ways.shape[0], r2.shape[0]))
	sqrB = np.broadcast_to(np.sum(np.power(r2, 2), 1).reshape(r2.shape[0], 1), (r2.shape[0], ways.shape[0])).transpose()
	#
	L2_dist_runway = np.sqrt(sqrA - 2*np.matmul(ways, r2.transpose()) + sqrB)
	#
	# print(L2_dist_runway.shape)
	L2_dist_runway = L2_dist_runway.squeeze().tolist()
	# print(L2_dist_runway)

	# print("trajs_x",trajs_x)
	xl_r1 = np.array(trajs_x) - 0.5
	xu_r1 = 2.5 - np.array(trajs_x)
	yl_r1 = np.array(trajs_y) + 2.5
	yu_r1 = -1.0 - np.array(trajs_y)
	zl_r1 = np.array(trajs_z) - 0.5
	zu_r1 = 0.7 - np.array(trajs_z)

	xl_r2 = np.array(trajs_x) - 2.5
	xu_r2 = 4.0 - np.array(trajs_x)
	yl_r2 = np.array(trajs_y) + 2.5
	yu_r2 = -0.08 - np.array(trajs_y)
	zl_r2 = np.array(trajs_z) - 0.3
	zu_r2 = 0.5 - np.array(trajs_z)
	# print("sub",xl_r1)
	# user_details = [{'name' : x, 'rank' : trajs_x.index(x)} for x in trajs_x]
	xl_r1 = list(enumerate(xl_r1))
	xu_r1 = list(enumerate(xu_r1))
	yl_r1 = list(enumerate(yl_r1))
	yu_r1 = list(enumerate(yu_r1))
	# zl_r1 = list(enumerate(zl_r1))
	# zu_r1 = list(enumerate(zu_r1))
	xl_r2 = list(enumerate(xl_r2))
	xu_r2 = list(enumerate(xu_r2))
	yl_r2 = list(enumerate(yl_r2))
	yu_r2 = list(enumerate(yu_r2))
	trajs_x = list(enumerate(trajs_x))
	trajs_y = list(enumerate(trajs_y))
	trajs_z = list(enumerate(trajs_z))

	phi_1u = list(enumerate(phi_1u))
	phi_1l = list(enumerate(phi_1l))
	phi_2u = list(enumerate(phi_2u))
	phi_2l = list(enumerate(phi_2l))
	phi_3u = list(enumerate(phi_3u))
	phi_3l = list(enumerate(phi_3l))
	# print('phi_3l',phi_3l)


	vel_x = list(enumerate(vels_x))
	vel_y = list(enumerate(vels_y))
	vel_z = list(enumerate(vels_z))
	# print('vel_z',vel_z)

	L2_runway = list(enumerate(L2_dist_runway))

	# print('trajs_z',trajs_z)
	# # stl
	spec = rtamt.STLDenseTimeSpecification(semantics=rtamt.Semantics.STANDARD)
	# spec = rtamt.STLSpecification(language=rtamt.Language.PYTHON)
	spec.name = 'runway r2'
	spec.declare_var('iq', 'float')
	spec.declare_var('up', 'float')
	spec.declare_var('x', 'float')
	spec.declare_var('xl_r1', 'float')
	spec.declare_var('xu_r1', 'float')
	spec.declare_var('yl_r1', 'float')
	spec.declare_var('yu_r1', 'float')
	spec.declare_var('zl_r1', 'float')
	spec.declare_var('zu_r1', 'float')

	spec.declare_var('xl_r2', 'float')
	spec.declare_var('xu_r2', 'float')
	spec.declare_var('yl_r2', 'float')
	spec.declare_var('yu_r2', 'float')
	spec.declare_var('y', 'float')
	spec.declare_var('yl', 'float')
	spec.declare_var('yu', 'float')
	spec.declare_var('z', 'float')

	spec.declare_var('zl', 'float')
	spec.declare_var('zu', 'float')
	spec.declare_var('final', 'float')

	spec.declare_var('out', 'float')

	spec.declare_var('phi_1u_head', 'float')
	spec.declare_var('phi_1l_head', 'float')

	spec.declare_var('phi_2u_head', 'float')
	spec.declare_var('phi_2l_head', 'float')

	spec.declare_var('phi_3u_head', 'float')
	spec.declare_var('phi_3l_head', 'float')

	# spec.set_var_io_type('xl', 'input')
	# spec.set_var_io_type('xu', 'input')

	spec.set_var_io_type('yl', 'input')
	spec.set_var_io_type('yu', 'input')

	spec.set_var_io_type('zl', 'input')
	spec.set_var_io_type('zu', 'input')

	spec.declare_var('vel_x', 'float')
	spec.declare_var('vel_y', 'float')
	spec.declare_var('vel_z', 'float')
	spec.declare_const('threshold_1', 'float', '3')

	spec.declare_const('threshold_2', 'float', '5')
	# spec.declare_var('x', 'float')
	# spec.declare_var('iq', 'float')
	# spec.declare_var('up', 'float')
	# spec.declare_var('xl', 'float')
	# spec.declare_var('xu', 'float')
	# spec.declare_var('y', 'float')
	# spec.declare_var('yl', 'float')
	# spec.declare_var('yu', 'float')
	# spec.declare_var('z', 'float')
	# spec.declare_var('zl', 'float')
	# spec.declare_var('zu', 'float')
	# spec.declare_var('out', 'float')
	# spec.declare_var('out', 'float')
	# spec.set_var_io_type('xl', 'input')
	# spec.set_var_io_type('xu', 'input')

	# spec.set_var_io_type('yl', 'input')
	# spec.set_var_io_type('yu', 'input')

	# spec.set_var_io_type('zl', 'input')
	# spec.set_var_io_type('zu', 'input')

	# spec.declare_var('phi_1', 'float')
	# spec.declare_var('phi_2', 'float')
	# spec.declare_var('phi_3', 'float')
	# # spec.set_var_io_type('out', 'output')
	# # spec.spec = '(eventually[0:0] ((always[0:0] (xl > 0.5)) and (always[0:0] (xu < 3.0))) )'
	# spec.add_sub_spec = ('phi_1 = (eventually[0:0] (always[0:0]( (xl > 0.5) and (xu < 2.5) ) and ( (yl > -2.5) and (yu < -1.0) ) and ( (zl > 0.5) and (zu < 0.7) ) ) )')
	# spec.add_sub_spec = ('phi_2 = (eventually[0:0] (always[0:0]( (xl > 2.5) and (xu < 4.0) ) and ( (yl > -2.5) and (yu < -0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')
	# spec.add_sub_spec = ('phi_3 = (eventually[0:0] (always[0:0]( (xl > 1.5) and (xu < 4.0) ) and ( (yl > -0.08) and (yu < 0.08) ) and ( (zl > 0.3) and (zu < 0.5) ) ) )')

	# spec.spec = '( ((phi_1) until[0:0] (phi_2)) until[0:0] (phi_3) )'

	# spec.spec = 'eventually(( ( (xl > 0.5) and (xu < 2.5) ) and ( (yl > -2.5) and (yu < -1.0) )  ) and eventually( ( (xl > 2.5) and (xu < 4.0) ) and ( (yl > -2.5) and (yu < -0.08) ) ) )'
	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08)) ))'

	# spec.spec =eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08)) )) 
	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -1.0)) and eventually( ((x > 1.5) and (x < 4.0) ) and ((y > -0.5) and (y < 0.2)) ) ))'
	spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.6) and (z < 0.8) ) and ( (vel_x > 0) and (vel_y < 0.5) )  and ( (phi_1l_head) and (phi_1u_head) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -1.0)) and ( (z > 0.4) and (z< 0.6) ) and ( (vel_x < 0.5) and (vel_y >0) )  and ( (phi_2l_head) and (phi_2u_head) ) and eventually((final < 0.2)  and  ((y > -0.5) and (y < 0.2)) and ( (vel_x < 0) and (vel_y < 0.5) )  and ( (phi_3l_head) and (phi_3u_head) ) ) ))'


	# spec.spec ='eventually( (( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) )) and eventually( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  and ((z > 0.3) and (z < 0.5)) ))'
	# spec.spec = '(eventually(always((( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) ))))) until 	(eventually(always(( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  and ((z > 0.3) and (z < 0.5)) ))))'
	# spec.spec = '(eventually(always((( (x >0.5) and (x < 2.5)) and ( (y > -2.5) and (y < -1.0) ) )))) until 	(eventually(always(( ((x > 2.5) and (x < 4.0) ) and ((y > -2.5) and (y < -0.08))  ))))'

	# spec.add_sub_spec('iq = x>0.5')
	# spec.add_sub_spec('up = x<2.5')
	# spec.spec = 'out = ((iq) until 	 (up) )'

	# spec.spec = 'eventually(( ( (x > 0.5) and (x < 2.5) ) and ( (y > -2.5) and (y < -1.0) ) and ( (z > 0.5) and (z < 0.7) ) ) and eventually( ( (x > 2.5) and (x < 4.0) ) and ( (y > -2.5) and (y < -0.08) ) and ( (z > 0.3) and (z < 0.5) ) ) )'
	try:
		spec.parse()
		# spec.pastify()
	except rtamt.STLParseException as err:
		print('STL Parse Exception: {}'.format(err))
		sys.exit()

	# out = spec.evaluate(['x', trajs_x],['y', trajs_y])
	out = spec.evaluate(['xl_r1', xl_r1],['xu_r1', xu_r1],['yl_r1', yl_r1],['yu_r1', yu_r1],['zl_r1', zl_r1],['zu_r1', zu_r1],['xl_r2', xl_r2],['xu_r2', xu_r2],['yl_r2', yl_r2],['yu_r2', yu_r2],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z],['x',trajs_x],['y',trajs_y],['z',trajs_z],['vel_x',vel_x],['vel_y',vel_y],['vel_z',vel_z],['final',L2_runway],['phi_1l_head',phi_1l],['phi_1u_head',phi_1u],['phi_2l_head',phi_2l],['phi_2u_head',phi_2u],['phi_3l_head',phi_3l],['phi_3u_head',phi_3u ])

	# out = spec.evaluate(['x', trajs_x],['y', trajs_y],['z', trajs_z],['xl', trajs_x],['xu',trajs_x],['yl', trajs_y],['yu',trajs_y],['zl', trajs_z],['zu',trajs_z])
	# print('Robustness offline: {}'.format(out))
	return out[0][1]

print(trajs10.shape)
# trajs10 = trajs10[:,:]
print(trajs10.shape)
num_traj = 256
with Parallel(n_jobs=4) as parallel:
    t = time.time()
    for i in range(10):
        parallel(delayed(monitor_R2)(trajs10) for i in range(num_traj))
print("time", time.time()-t)

t = time.time()

for i in range(10):
    Parallel(n_jobs=4)(delayed(monitor_R2)(trajs10) for i in range(num_traj))
print("time", time.time()-t)
t = time.time()
for i in range(10):
    for i in range(num_traj):
        (monitor_R2(trajs10))
print("time", time.time()-t)

