import numpy as np
import time
import yaml
import argparse
from copy import deepcopy
from env import Biped2dBullet; 


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode",type=str, default='jog', help="specify mode of locomotion to run from config file")
parser.add_argument("-f", "--file_path", type=str, default="settings/config.yml", help="path to config file (yml format)")
parser.add_argument("-t", "--sim_dt", type=float, default=0.0004, help="simulation timestep")
parser.add_argument("-i", "--init_index", type=int, default=0, help="starting index of the FSM")
parser.add_argument("-mi", "--max_iters", type=int, default=2e4, help="iterations of episode")
parser.add_argument("-s", "--save_trajectory", action='store_true', default=False, help="whether to save the motion trajectory (needed for imitation learning)")
parser.add_argument("-sp", "--save_path", type=str, default="", help="save_path of motion trajectory")
args = parser.parse_args()
mode = args.mode

filename = args.file_path
with open(filename, 'r') as f:
    f_data = yaml.load(f)
settings = f_data[mode]
curr_index = args.init_index
foot_iters = int(settings['dt']/args.sim_dt)

max_iters = args.max_iters

env = Biped2dBullet(); 
env.dt = args.sim_dt
env.mu = settings['mu']
env.max_torque = settings['max_torque']
env.max_iters = max_iters
env.g = settings['g']
env.kp = settings['kp']
env.kd = settings['kd']


angles = settings['targets']
#torso control and swing placement params 
kp_torso = settings['kp_torso']
kd_torso = settings['kd_torso']
c_d = settings['c_d']
c_v = settings['c_v']

if args.save_trajectory:
    traj = []
def get_state_index(state, curr_index, iters_this_state, foot_iters):
    """Returns the next state in the FSM model
    """
    if curr_index==0 or curr_index==2:
        if iters_this_state>foot_iters:
            curr_index+=1
    elif curr_index==1:
        #check if right foot made ground contact
        if abs(state[-2]-1.)<1e-5:
            curr_index+=1
    else:
        if abs(state[-1]-1.)<1e-5:
            curr_index=0
    return curr_index    

iters_this_state=0
state = env.reset()

for i in range(int(max_iters)):
    targ_ang = deepcopy(angles[curr_index])
    action = np.concatenate([np.array(targ_ang[1:]), np.zeros(6)])  
     
    #BALANCE FEEDBACK
    if (abs(state[-2]-1.)<1e-5):
        #right foot on ground
        swing_ind = 3
        stance_ind = 0
        if curr_index==3: 
            d = state[14+0*(6)+1] - state[14+3*(6)+1] 
            v =  state[14+0*6+4]
            action[3] = action[3] + c_v*v +c_d*d
        
    elif (abs(state[-1]-1.)<1e-5 or i==0):
        #left foot on ground
        swing_ind= 0
        stance_ind = 3
        if curr_index==1: 
            d = state[14+0*(6)+1] - state[14+6*(6)+1] 
            v = state[14+0*6+4]
            action[0] = action[0] + c_v*v +c_d*d 
    #no pd target for stance hip     
    action[stance_ind] = np.nan
    
    #TORSO CONTROL    
    theta_torso = state[0]
    omega_torso = state[1]
    torque_stance = -kp_torso*(theta_torso-targ_ang[0]) - kd_torso*omega_torso #-0.065
    torque_stance += -env.kp[swing_ind]*(state[2*(swing_ind+1)]-targ_ang[swing_ind+1]) - env.kd[swing_ind]*state[2*(swing_ind+1)+1]
    action[6+stance_ind] = -torque_stance  
     
    state, reward, done, _ = env.step(action)
    env.render()
    if args.save_trajectory:
        traj.append(np.array([state,action]))
    iters_this_state+=1
    next_ind = get_state_index(state,curr_index, iters_this_state, foot_iters = foot_iters)
    if next_ind!=curr_index:
        iters_this_state=0
    curr_index = next_ind

if args.save_path is not "":
    np.save(args.save_path, np.array(traj))

