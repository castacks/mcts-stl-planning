from costmap import CostMap
from matplotlib import pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# import seaborn as sns


def plot_cost():
    mp3d = CostMap('./dataset/111_days/processed_data/train')
    fig = plt.figure()
    sp = fig.add_subplot(111)
    # fig.show()
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print(x.shape)
    xv, yv = np.meshgrid(x, y,sparse=False)
    arr = np.empty((120,120))
    for i in tqdm(range(len(xv))):
        for j in range(len(yv)):

            z = 0.5096 #(km)
            angle = -0 #degrees
            wind = 1 # 1 for right -1 for left
            # print(xv.shape)
            
            # print(mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind))
            # plt.scatter(x= xv[i,j], y = yv[i,j], c = mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind))
            arr[i][j] = mp3d.state_value(xv[i,j], yv[i,j], z, angle, wind)
    # plt.colorbar()
    xlabels = ['{:3.1f}'.format(x) for x in xv[0,:]]

    ax = sns.heatmap(arr,xticklabels=xlabels, yticklabels=xlabels)
    ax.invert_yaxis() 
    ax.set_xticks(ax.get_xticks()[::3])
    ax.set_xticklabels(xlabels[::3])
    ax.set_yticks(ax.get_yticks()[::3])
    ax.set_yticklabels(xlabels[::3])
    plt.savefig("mcts_"+ ".png")

def generate_ref():
    x = np.arange(-6.0,3,43.321/1000)
    y = np.repeat(-2,len(x))
    z = np.repeat(-0.5,len(x))
    print(len(x))
    traj = np.vstack((x,y,z)).transpose()
    print("traj",traj.shape)
    x = np.array([[ 0.0 ,43.32168421052632 ,86.64336842105263 ,129.96505263157894 ,173.28673684210526 ,216.6084210526316 ,259.93010526315794 ,303.2517894736842 ,346.5734736842105 ,389.89515789473677 ,433.2168421052632 ,476.5385263157894 ,519.8602105263158, 563.1818947368421, 606.5035789473684, 649.8252631578947, 693.146947368421, 736.4686315789475 ,779.7903157894735 ,823.1120000000001 ],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]])
    x[0] = x[0] + 627.0
    x[0] = x[0]/1000
    x[1] = x[1] - 2.5
    x[2] = x[2] + 0.5
    curr_position = x.transpose()

    print(traj.shape,curr_position.shape)
    # print(np.linalg.norm(traj-np.tile(curr_position[0,:],(len(y),1)),axis=1))
    idx_closest = np.argmin(np.linalg.norm(traj-np.tile(curr_position[0,:],(len(y),1)),axis=1))
    idx_closest = 207
    print(min(idx_closest+20,traj.shape[0]-2),traj[207,:])

if __name__ == '__main__':
    generate_ref()
