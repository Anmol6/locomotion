import os

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data





MAX_TORQUE=2e3
dt=0.001
GRAVITY=-9.8
mu = 0.65
kp = np.array([ 0. ,  0.3,  0.2,  0.1,  0.3,  0.2,  0.1])
kd = 0.1*kp
max_iters = 5e4
target_vel = 0.#3.5
class Biped2dBullet(gym.Env):
    """2D Biped environment 
    """
    def __init__(self):
        self.p = p
        #self.physics_client = self.p.connect(self.p.GUI)
        self.num_control_steps=1
        #self.action_space = spaces.Box(low=-MAX_TORQUE*np.ones(12), high=MAX_TORQUE*np.ones(12))
        #self.observation_space = spaces.Box(low=-np.inf*np.ones(58), high=np.inf*np.ones(58))
        self.kp = kp
        self.kd = kd
        self.joint_idx = [2,3,4,5,6,7,8]
        self.max_torque=MAX_TORQUE
        self.scale = 1.
        self.dt = dt
        self.g = GRAVITY
        self.mu = mu
        self.ground_kp=1e6
        self.ground_kd=6e3
        self.plane_ids = [] 
        self.target_vel = target_vel
        self.max_iters = max_iters
    
    def reset(self, disable_gui=False):
        if disable_gui:
            self.physics_client = self.p.connect(self.p.DIRECT)
        elif disable_gui==False:
            self.physics_client = self.p.connect(self.p.GUI)
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0,0,self.g)
        #load character
        cubeStartPos = [0,0,1.185]
        cubeStartOrientation = self.p.getQuaternionFromEuler([0.,0.,0.])
        self.sim_id = self.p.loadURDF("assets/biped2d.urdf", cubeStartPos, cubeStartOrientation)
        #load terrain
        self.plane_id = self.p.loadSDF("assets/plane_stadium.sdf")[0]
        self.p.setGravity(0,0,GRAVITY)
        self.plane_ids.append(self.plane_id)
        self.iters = 0
        for plane_id in self.plane_ids:
            self.p.changeDynamics(plane_id, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd) 
        
        #disable motors (in order to do control via torques)
        for joint in range(p.getNumJoints(self.sim_id)):
            self.p.setJointMotorControl2(self.sim_id, joint, p.VELOCITY_CONTROL, force=0)
        
        for i in range(250):#+ 10*np.random.randint(low=0, high=20)):
            self.p.stepSimulation()
    
        return self.get_state() 
    
    def make_step(self, step_x, step_y,step_z):
        plane_id = self.p.loadURDF("plane.urdf", [0,step_y-15,step_z],self.p.getQuaternionFromEuler([0,0,0]))
        plane_id2 = self.p.loadURDF("plane.urdf", [0,step_y+0.48,step_z-15],self.p.getQuaternionFromEuler([-1.54,0,0]))
        self.plane_ids.append(plane_id)
        self.plane_ids.append(plane_id2)
   
    def has_contact(self, bullet_client, bodyA, bodyB, linkA):
        if len(bullet_client.getContactPoints(bodyA,bodyB, linkIndexA=linkA))==0:
            return False
        else:
            return True

    def step(self, action):
        """Performs PD control and applies additional torque for the 6 joints
        """
        for i in range(self.num_control_steps):
            #pd control
            self.pd_control(action[:6])
            #additional torque
            self.apply_torque(action[6:])   
        state = self.get_state()
        reward = self.get_reward()
        done = False
        if self.p.getLinkState(self.sim_id,2)[0][2]<0.5:
            #abs(state[18])<0.05 and self.iters>2000 and abs(self.target_vel)>0.05: 
            done = True
            reward+=-100000
        if self.iters>self.max_iters:# or (self.iters>2000 and abs(state[18])<0.05 and abs(self.target_vel)>0.05):
            done = True
        self.iters+=1
        return state, reward, done, None 

    def render(self):
        self.p.resetDebugVisualizerCamera(2,90,3, [self.p.getLinkState(self.sim_id,2)[0][0], self.p.getLinkState(self.sim_id,2)[0][1],0.8]) 

    def pd_control(self, action):
        """Performs PD control for target angles (given by action)
        """
        for i,j in enumerate(self.joint_idx[1:]):
            if action[i] is not None and not np.isnan(action[i]):
                torque_i = -self.kp[i]*(self.p.getJointState(self.sim_id, j)[0]-action[i]) - self.kd[i]*self.p.getJointState(self.sim_id, j)[1]
                self.p.setJointMotorControl2(self.sim_id, j, self.p.TORQUE_CONTROL, force=np.clip(self.scale*torque_i, -self.max_torque,self.max_torque))
        self.p.stepSimulation()
    
    def apply_torque(self, action):
        """Applies given torque at each joint
        """
        for i,j in enumerate(self.joint_idx[1:]):
            if abs(action[i])>1e-5:
                self.p.setJointMotorControl2(self.sim_id, j, self.p.TORQUE_CONTROL, force=np.clip(self.scale*action[i],-self.max_torque,self.max_torque))
        self.p.stepSimulation()
    
    def get_reward(self):
        """Reward is the squared difference between torso velocity and target forwards velocity
        """
        return -(self.target_vel - self.p.getLinkState(self.sim_id, 2, computeLinkVelocity=1)[-2][1])**2
    
    def get_state(self):
        """Returns angular position and angular velocity of every joint, position and velocity of every link, and foot contact information
        """
        
        state = []
        for j in self.joint_idx:
            joint_state = self.p.getJointState(self.sim_id,j)
            state.append(joint_state[0])
            state.append(joint_state[1])
        for j in range(2,9):
            link_state = self.p.getLinkState(self.sim_id, j, computeLinkVelocity=1)
            state.append(link_state[0][0])
            state.append(link_state[0][1])
            state.append(link_state[0][2])
            state.append(link_state[-2][0])
            state.append(link_state[-2][1])
            state.append(link_state[-2][2])

        for foot_link in [5,8]:
            if self.has_contact(self.p, self.sim_id, self.plane_id, foot_link):
                state.append(1.)
            else:
                state.append(-1.)
        return np.array(state)
    def close(self):
        self.p.disconnect()
