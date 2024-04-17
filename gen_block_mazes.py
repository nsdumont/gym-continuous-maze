import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) #5 
sizes = np.arange(5,20)
all_mats = []
fig, axs = plt.subplots(3,5)
axs = axs.flatten()
for i, size in enumerate(sizes):
    randmat = np.random.choice(2, size=(size,size), p = [0.2,0.8])
    start = np.array([0,0])
    end = np.array([size-1, size-1])
    randmat[start[0],start[1]] = 0
    randmat[end[0],end[1]] = 0
    
    pos = start
    go = np.any(pos != end)
    action = 1
    while go:
        next_pos = pos + np.array([[0,-1],[1,0],[0,1],[-1,0]]) 
        
        dists = np.linalg.norm(next_pos-end[None,:], axis=1)
        ps = np.ones(4)
        ps[np.argmin(dists)] = 1.3
        ps[action] = 1.1
        ps = ps*np.array([pos[1]>0, pos[0] < size-1, pos[1] < size-1, pos[0]>0])
        action = np.random.choice(4, size = 1, p = ps/sum(ps))
        if action == 0:
            pos[1] = pos[1] - 1
        elif action == 1:
            pos[0] = pos[0] + 1
        elif action==2:
            pos[1] = pos[1] + 1
        else:
            pos[0] = pos[0] - 1
        randmat[pos[0],pos[1]] = 0
        go = np.any(pos != end)
    # randmat = np.ones((size,size)) 
    # randmat[1:-1,1:-1] = np.random.choice(2, size=(size-2,size-2), p = [0.3,0.7])
    # start = np.array([1,1])
    # end = np.array([size-2, size-2])
    # randmat[start[0],start[1]] = 0
    # randmat[end[0],end[1]] = 0
    
    # pos = start
    # go = np.any(pos != end)
    # while go:
    #     next_pos = pos + np.array([[0,-1],[1,0],[0,1],[-1,0]]) 
        
    #     dists = np.linalg.norm(next_pos-end[None,:], axis=1)
    #     ps = np.ones(4)
    #     ps[np.argmin(dists)] = 1.5
    #     ps = ps*np.array([pos[1]>1, pos[0] < size-2, pos[1] < size-2, pos[0]>1])
    #     action = np.random.choice(4, size = 1, p = ps/sum(ps))
    #     if action == 0:
    #         pos[1] = pos[1] - 1
    #     elif action == 1:
    #         pos[0] = pos[0] + 1
    #     elif action==2:
    #         pos[1] = pos[1] + 1
    #     else:
    #         pos[0] = pos[0] - 1
    #     randmat[pos[0],pos[1]] = 0
    #     go = np.any(pos != end)
    
    all_mats.append(randmat)
    axs[i].imshow(randmat)
  
    dir = '/home/ns2dumon/Documents/Github/gym-continuous-maze/gym_continuous_maze/maze_samples/'
for k, size in enumerate(sizes):
    save_mat = []
    save_mat.append(np.array([[0,size],[0,0]]))
    save_mat.append(np.array([[0,0],[0,size]]))
    save_mat.append(np.array([[size,size],[0,size]]))
    save_mat.append(np.array([[0,size],[size,size]]))
    for i in range(size):
        for j in range(size):
            if all_mats[k][i,j] == 1:
                save_mat.append(np.array([[i,i+1],[j,j+1]]))
    
    np.save(dir + f'block2d_{size}x{size}.npy', np.array(save_mat), allow_pickle=False, fix_imports=True)
        
# mystr = ""
# for k, size in enumerate(sizes):
#     mystr += f"class ContinuousMazeBlocks{size}x{size}(ContinuousMaze):\n"
#     mystr += "\tdef __init__(self, enable_render=False,render_mode=None):\n"
#     mystr += f"\t\tsuper(ContinuousMazeBlocks{size}x{size}, self).__init__(maze_file='block2d_{size}x{size}.npy',render_mode=render_mode)\n"
#     mystr += "\n"



# mystr2 = ""
# for k, size in enumerate(sizes):
#     max_steps = size*size*20
#     mystr2 += "register(\n"
#     mystr2 += f"\tid='ContinuousMazeBlocks-{size}x{size}-v0',\n"
#     mystr2 += f"\tentry_point='gym_continuous_maze.gym_continuous_maze:ContinuousMazeBlocks{size}x{size}',\n"
#     mystr2 += f"\tmax_episode_steps={max_steps},\n"
#     mystr2 += ")\n\n"

