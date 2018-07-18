import multiprocessing
import numpy as np
import time
import yaml
import argparse
import gc

from deap import base
from deap import creator
from deap import tools
from deap import cma
from env import Biped2dBullet; 
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("-lm", "--load_mode",type=str, default='jog', help="specify mode of locomotion to run from config file")
parser.add_argument("-lp", "--load_path", type=str, default="settings/config.yml", help="path to .yml config file where initial parameters are stored")
parser.add_argument("-sm", "--save_mode",type=str,  help="specify name of mode of saved parametrs")
parser.add_argument("-sp", "--save_path", type=str, help="path to .yml config file where optimized parameters are stored")
parser.add_argument("-tv", "--target_vel", type=float, default=-0.5, help="target velocity used in optimization")
parser.add_argument("-hm", "--hip_movement", type=float, default=1., help="how much hip movement should the final motion have (0 to 1.)")
parser.add_argument("-g", "--gait_period", type=float, default=.5, help="how long should the gait be in the final motion(0 to 1.)")

parser.add_argument("-ng", "--n_gen", type=int, default=500, help="number of iterations of the algorithm")
parser.add_argument("-np", "--n_population", type=int, default=100, help="number of samples evaluated at each iteration")
parser.add_argument("-nproc", "--n_processes", type=int, default=14, help="number of threads for cma")
args = parser.parse_args()

load_mode = args.load_mode
load_path = args.load_path

save_mode = args.save_mode 
save_path = args.save_path  
load_name = 'walk0_from_jog_symmetric_pdparams_'



test_mode = False#if True, parameters in settings/run_name.npy are used to run environment in a GUI. No optimization is done

#desired velocity for optimization
target_vel = args.target_vel
#movement style parameters, each from 0 to 1.; these are translated into parameter constriants for cma
hip_movement = args.hip_movement
gait_period = args.gait_period

with open(load_path, 'r') as f:
    f_data = yaml.load(f)
settings = f_data[load_mode]

def evaluate_actor(params):
    f = 0.
    dt = 4e-4#0.0004
    max_iters = int(2.4e4)
    env = Biped2dBullet(); 
    env.dt = dt
    env.max_iters = max_iters
    env.mu = settings['mu']    
    env.max_torque = settings['max_torque']
    env.g = settings['g']
    env.kp = [settings['kp'][i]+100*params[1][15] for i in range(len(settings['kp']))]
    env.kd = 0.1*np.array(settings['kp'])#settings['kd'] 
    env.target_vel = target_vel

    #angles = params[1][:7]
    fsm1 = params[1][:7]
    fsm2 = params[1][7:14]
    fsm3 = [fsm1[0], fsm1[4], fsm1[5], fsm1[6], fsm1[1], fsm1[2], fsm1[3]]
    fsm4 = [fsm2[0], fsm2[4], fsm2[5], fsm2[6], fsm2[1], fsm2[2], fsm2[3]]

    angles = [fsm1, fsm2, fsm3, fsm4]
    #torso control and swing placement params 
    kp_torso = max(100*fsm1[4] + settings['kp_torso'],10.)
    kd_torso = max(100*fsm2[4]  + settings['kd_torso'],1.)
    c_d = settings['c_d']
    c_v = settings['c_v']

    foot_iters = int((settings['dt']+params[1][14])/dt) 
    #int(params[1][14]/dt)#int(settings['dt']/dt)

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

    
    
    for curr_index in [0,2]:
        #if(curr_index==0):
        state = env.reset(disable_gui = not test_mode)
        
        iters_this_state=0
        for i in range(max_iters):
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
            
            if(test_mode):
                env.render()
                time.sleep(dt)
                print(state[18])
            
            f+=reward
            if done:
                break;
            iters_this_state+=1
            next_ind = get_state_index(state,curr_index, iters_this_state, foot_iters = foot_iters)
            if next_ind!=curr_index:
                    iters_this_state=0
            curr_index = next_ind
        
        env.close()
    del env.p
    del env
    gc.collect()

    return (f,)
# Define parameter types (list) and fitness types (weighted sum of 1 scalar)
# For multiprocessing, this has to be outside of main()
if (test_mode==False):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

#import ipdb
def save_params(model_params):
    #import ipdb
    #ipdb.set_trace()
    t1 = model_params[:7]
    t2 = model_params[7:14]
    kp_torso = max(100*t1[4] + settings['kp_torso'],10.)
    kd_torso = max(100*t2[4]  + settings['kd_torso'],1.)
    kp = [settings['kp'][i]+100*model_params[15] for i in range(len(settings['kp']))]
    kd = [0.1*kp[i] for i in range(len(kp))]
    f = {
            save_mode:{
            'targets':[t1, t2, [t1[0], t1[4], t1[5], t1[6], t1[1], t1[2], t1[3]], [t2[0], t2[4], t2[5], t2[6], t2[1], t2[2], t2[3]] ],
            'kp_torso': kp_torso,
            'kd_torso': kd_torso,
            'kp': kp,
            'kd':kd,
            'dt': settings['dt'] + model_params[14],
            'c_d': settings['c_d'],
            'c_v': settings['c_v'],
            'mu': settings['mu'],
            'max_torque': settings['max_torque'],
            'g': settings['g']
            }
        }
    
    with open(save_path, 'w') as outfile:
        yaml.dump(f, outfile, default_flow_style=False)

def main():
    env_name = "SomeEnv-v0"
    n_population  = args.n_population
    n_gen = args.n_gen 
    
    model_params = []
    for el in settings['targets'][0]:
        model_params.append(el)
    
    for el in settings['targets'][1]:
        model_params.append(el)
    model_params[4] =0. #1.*settings['kp_torso'] 
    model_params[11] =0.# 1.*settings['kd_torso']
    model_params.append(0.0)#settings['dt'])
    model_params.append(0.0)

    model_params = np.load('data/' + load_name+'.npy')
    print('initial parameters')
    print(model_params)
    
    if(test_mode==False):
        strategy = cma.Strategy(centroid=model_params, sigma=.05, lambda_=n_population)
        
            
        # Initialize toolbox and multiprocessing
        toolbox = base.Toolbox()
        toolbox.register("update", strategy.update)
        toolbox.register("evaluate", evaluate_actor)
        pool = multiprocessing.Pool(processes=args.n_processes)
        toolbox.register("map", pool.map)


        # Custom generate function to scale the variance for different params
        toolbox.register(
        "generate",
        strategy.generate,
        creator.Individual,
        )
        
        # Logging utilities
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
            
        for gen in range(n_gen):
            # Generate a new population
            population = toolbox.generate()
            for j in range(len(population)):
                for i in range(len(model_params)):
                    if i==4:
                        min_val = -1
                        max_val = 1
                    if i==11:
                        min_val = -1
                        max_val = 1.
                    if i==0 or i==7:
                        min_val = -0.3
                        max_val = 0.3
                    if i==1 or i==8:
                        min_val = -2.+(1-hip_movement)
                        max_val = 2. - (1-hip_movement)
                    if i==2 or i==5 or i==9 or i==12:
                        min_val = -2.
                        max_val = .5
                    if i==3 or i==6 or i==10 or i==13:
                        min_val = -1.
                        max_val = 1.
                    if (i==14):
                        
                        min_val = -0.05 + (gait_period - 0.5)/10.
                        max_val=0.05 + (gait_period - 0.5)/10.
                    if (i==15):
                        min_val = -1.
                        max_val=1.
                    population[j][i] = np.clip(population[j][i], min_val, max_val)
                
            fitnesses = toolbox.map(toolbox.evaluate, zip([env_name]*len(population), population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            if hof is not None:
                hof.update(population)

            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            print(logbook.stream)
            print(hof[0])
            if(gen%4==0):
                save_params(np.array(hof[0]).tolist())
                #np.save('data/' + str(run_name) +'_' + str(hip_movement) + '_' + str(gait_period) + '.npy', np.array(hof[0]))
    else:
        evaluate_actor([None, model_params])
if __name__ == "__main__":
	main()

