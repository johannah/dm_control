

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dm_env
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control import robot
from IPython import embed
from copy import deepcopy
import numpy as np

# the kinova jaco2 ros exposes the joint state at ~52Hz
# CONTROL_TIMESTEP should be long enough that position controller can also reach the destination in mujoco if it is reacher. Tested with .1 CONTROL_TIMESTEP and .1 maximum relative deg step in 7DOF jaco for position controllers (with tuned gain).
_CONTROL_TIMESTEP = .1
_POSITION_EPISODE_TIME_LIMIT = 2 # for position controller - these seconds divided by thethe timestep specified in the xml file (.02) is the number of steps per episode
_LONG_POSITION_EPISODE_TIME_LIMIT = 20 # for position controller - we the time limit doesnt really make sense
_LONG_EPISODE_TIME_LIMIT = 20
_SHORT_EPISODE_TIME_LIMIT = 10
_TINY_EPISODE_TIME_LIMIT = 5
_BIG_TARGET = .05
_SMALL_TARGET = .015
# 7DOF Jaco2
#D1 Base to shoulder 0.2755
#D2 First half upper arm length 0.2050
#D3 Second half upper arm length 0.2050
#D4 Forearm length (elbow to wrist) 0.2073
#D5 First wrist length 0.1038
#D6 Second wrist length 0.1038
#D7 Wrist to center of the hand 0.1600
#e2 Joint 3-4 lateral offset 0.0098

# Params for Denavit-Hartenberg Reference Frame Layout (DH)
DH_lengths =  {'D1':0.2755, 'D2':0.2050, 
                    'D3':0.2050, 'D4':0.2073,
                    'D5':0.1038, 'D6':0.1038, 
                    'D7':0.1600, 'e2':0.0098}

# DH transform from joint angle to XYZ from kinova robotics ros code
DH_a = (0, 0, 0, 0, 0, 0, 0)
DH_d = (-DH_lengths['D1'], 
         0, 
         -(DH_lengths['D2']+DH_lengths['D3']), 
         -DH_lengths['e2'], 
         -(DH_lengths['D4']+DH_lengths['D5']), 
         0, 
         -(DH_lengths['D6']+DH_lengths['D7']))

DH_alpha = (np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi)
DH_theta_sign = (1, 1, 1, 1, 1, 1, 1)
DH_theta_offset = (np.pi,0.0, 0.0, 0.0, 0.0,0.0,np.pi/2.0)


# size of target in meters
SUITE = containers.TaggedTasks()

def DHtransformEL(d,theta,a,alpha):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                  [0.0, np.sin(alpha), np.cos(alpha),d],
                  [0.0,0.0,0.0,1.0]])
    return T

def trim_and_check_pose_safety(position, fence):
    """
    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
    """
    hit = False
    safe_position = []
    for ind, dim in enumerate(['x','y','z']):
        if max(fence[dim]) < position[ind]:
            out = max(fence[dim])
            hit = True
            print('hit max: req {} is more than fence {}'.format(position[ind], max(fence[dim])))
        elif position[ind] < min(fence[dim]):
            out = min(fence[dim])
            hit = True
            print('hit min: req {} is less than fence {}'.format(position[ind], min(fence[dim])))
        else:
            out = position[ind]
        safe_position.append(out)
    return safe_position, hit

def get_model_and_assets(xml_name):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(xml_name), common.ASSETS

@SUITE.add('benchmarking', 'random_relative_position_reacher_7DOF')
def random_relative_position_reacher_7DOF(random=None, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, robot_server_ip='127.0.0.1', robot_server_port=9030, physics_type='mujoco', environment_kwargs={}):
    xml_name='jaco_j2s7s300_position.xml'
    robot_name = 'j2s7s300'
    start_position='random'
    fully_observable=True 
    action_penalty=True
    relative_step=True 
    relative_rad_max=.1
    fence=fence 
    degrees_of_freedom=7 
    extreme_joints=[4,6,7] 
    target_size=_BIG_TARGET 
    target_type='random' 
    #control_timestep=_CONTROL_TIMESTEP
    episode_timelimit=_POSITION_EPISODE_TIME_LIMIT 
    if physics_type == 'robot':
        physics = RobotPhysics()
        physics.initialize(robot_name, robot_server_ip, robot_server_port, fence)
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(robot_name, random, degrees_of_freedom, fence)

    safety_physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    safety_physics.initialize(robot_name, random, degrees_of_freedom, fence)
    task = Jaco(safety_physics=safety_physics, random=random, start_position=start_position, 
             fully_observable=fully_observable, 
             action_penalty=action_penalty,
             relative_step=relative_step, 
             relative_rad_max=relative_rad_max,  
             fence=fence,
             degrees_of_freedom=degrees_of_freedom, 
             extreme_joints=extreme_joints, 
             target_size=target_size, 
             target_type=target_type, 
             )
    env = control.Environment(
            physics, task, 
            n_sub_steps=1,
            time_limit=episode_timelimit, 
            **environment_kwargs)
    return env


@SUITE.add('benchmarking', 'relative_position_reacher_7DOF')
def relative_position_reacher_7DOF(random=None, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, robot_server_ip='127.0.0.1', robot_server_port=9030, physics_type='mujoco', environment_kwargs={}):
    xml_name='jaco_j2s7s300_position.xml'
    robot_name = 'j2s7s300'
    start_position='home'
    fully_observable=True 
    action_penalty=True
    relative_step=True 
    relative_rad_max=.1
    fence=fence 
    degrees_of_freedom=7 
    extreme_joints=[4,6,7] 
    target_size=_BIG_TARGET 
    target_type='random' 
    episode_timelimit=_POSITION_EPISODE_TIME_LIMIT 
    if physics_type == 'robot':
        physics = RobotPhysics()
        physics.initialize(robot_name, robot_server_ip, robot_server_port, fence)
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(robot_name, random, degrees_of_freedom, fence)

    safety_physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    safety_physics.initialize(robot_name, random, degrees_of_freedom, fence)
    task = Jaco(safety_physics=safety_physics, random=random, start_position=start_position, 
             fully_observable=fully_observable, 
             action_penalty=action_penalty,
             relative_step=relative_step, 
             relative_rad_max=relative_rad_max,  
             fence=fence,
             degrees_of_freedom=degrees_of_freedom, 
             extreme_joints=extreme_joints, 
             target_size=target_size, 
             target_type=target_type, 
             )
    env = control.Environment(
            physics, task, 
            n_sub_steps=1,
            time_limit=episode_timelimit, 
            **environment_kwargs)
    return env

@SUITE.add('benchmarking', 'fixed_relative_position_reacher_7DOF')
def fixed_relative_position_reacher_7DOF(random=None, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, target_type='fixed', fixed_target_position=[.2,-.2,.5], robot_server_ip='127.0.0.1', robot_server_port=9030, physics_type='mujoco', environment_kwargs={}):
    xml_name='jaco_j2s7s300_position.xml'
    robot_name = 'j2s7s300'
    start_position='home'
    fully_observable=True 
    action_penalty=True
    relative_step=True 
    relative_rad_max=.1
    fence=fence 
    degrees_of_freedom=7 
    extreme_joints=[4,6,7] 
    target_size=_BIG_TARGET 
    episode_timelimit=_POSITION_EPISODE_TIME_LIMIT 
    if physics_type == 'robot':
        physics = RobotPhysics()
        physics.initialize(robot_name, robot_server_ip, robot_server_port, fence)
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(robot_name, random, degrees_of_freedom, fence)

    safety_physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    safety_physics.initialize(robot_name, random, degrees_of_freedom, fence)
    task = Jaco(safety_physics=safety_physics, random=random, start_position=start_position, 
             fully_observable=fully_observable, 
             action_penalty=action_penalty,
             relative_step=relative_step, 
             relative_rad_max=relative_rad_max,  
             fence=fence,
             degrees_of_freedom=degrees_of_freedom, 
             extreme_joints=extreme_joints, 
             target_size=target_size, 
             target_type=target_type, 
             fixed_target_position=fixed_target_position, 
             )
    env = control.Environment(
            physics, task, 
            n_sub_steps=1,
            time_limit=episode_timelimit, 
            **environment_kwargs)
    return env



@SUITE.add('benchmarking', 'configurable_reacher')
def configurable_reacher(xml_name='jaco_j2s7s300_position.xml', 
                         random=None, 
                         start_position='home', 
                         fully_observable=True, 
                         action_penalty=True, 
                         relative_step=True, 
                         relative_rad_max=.1, 
                         fence = {'x':(-1.5,1.5),'y':(-1.5,1.5),'z':(-1.5,1.5)}, 
                         degrees_of_freedom=7, 
                         extreme_joints=[4,6,7], 
                         target_size=_BIG_TARGET, 
                         target_type='random', 
                         fixed_target_position=[.2,-.2,.5], 
                         robot_name = 'j2s7s300',
                         robot_server_ip='127.0.0.1', 
                         robot_server_port=9030, 
                         physics_type='mujoco', 
                         control_timestep=_CONTROL_TIMESTEP, 
                         episode_timelimit=_LONG_POSITION_EPISODE_TIME_LIMIT, 
                         environment_kwargs={}):
    if physics_type == 'robot':
        physics = RobotPhysics()
        physics.initialize(robot_name, robot_server_ip, robot_server_port, fence)
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(robot_name, random, degrees_of_freedom, fence)

    safety_physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    safety_physics.initialize(robot_name, random, degrees_of_freedom, fence)
    """ configurable Jaco """
    task = Jaco(
             safety_physics=safety_physics,
             random=random,
             start_position=start_position, 
             fully_observable=fully_observable, 
             action_penalty=action_penalty,
             relative_step=relative_step, 
             relative_rad_max=relative_rad_max,  
             fence=fence,
             degrees_of_freedom=degrees_of_freedom, 
             extreme_joints=extreme_joints, 
             target_size=target_size, 
             target_type=target_type, 
             fixed_target_position=fixed_target_position)
 
 
    #control_timestep=control_timestep,
    env = control.Environment(
            physics, task, 
            n_sub_steps=1,
            time_limit=episode_timelimit, 
            **environment_kwargs)
  
    return env
def _find_joint_coordinate_extremes(major_joint_angles):  
    """calculate xyz positions for joints form cartesian extremes
    major_joint_angles: ordered list of joint angles in radians (len 7 for 7DOF arm)"""
    """ July 2020 - 
    JRH sanity checked these values by setting the real 7DOF Jaco 2 robot to the "home position" 
    and measured the physical robot against the DH calculations of extreme joints.
    In this function, we care about the elbow (joint 4), wrist, (joint6) and tool pose (fingertips)

    home_joint_angles = np.array([4.92,    # 283 deg
                                  2.839,   # 162.709854126
                                  0.,       # 0 
                                  .758,    # 43.43
                                  4.6366,  # 265.66
                                  4.493,   # 257.47
                                  5.0249,  # 287.9
    
    Unfortunately, I only have an Imperial tape measure, so converting to inches first:
    ---
    index   xm         ym       zm      xin       yin       zin        measured_description
    6       -.304     -.149     .504    -11.9685  -5.866    19.685     finger tips!  this is appropriate to use for tool pose
    5       -.04      -.144     .515    1.57      5.66      20.27      middle of joint 6 starting with 1 joint indexing
    3       .016      .122      .667    .629      4.8       26.25      joint 4 starting with 1 joint indexing
    """
    extreme_xyz = []
    major_joint_angles = major_joint_angles[:len(DH_theta_sign)]
    # transform first!
    Tall = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float)
    for i, angle in enumerate(major_joint_angles):
        DH_theta = DH_theta_sign[i]*angle + DH_theta_offset[i]
        T = DHtransformEL(DH_d[i], DH_theta, DH_a[i], DH_alpha[i])
        Tall = np.dot(Tall, T)
        #if i+1 in extreme_joints:
        extreme_xyz.append([Tall[0,3], Tall[1,3], Tall[2,3]])
    extremes = np.array(extreme_xyz)
    return extremes


class MujocoPhysics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def initialize(self, robot_name='j2s7s300', seed=None, degrees_of_freedom=7, fence={'x':(-.5,.5), 'y':(-1,.4), 'z':(.15,1)}, control_type='position'):
        # only compatible with j2
        robot_model = robot_name[:2]
        assert robot_model == 'j2'
        # only tested with 7dof, though 6dof should work with tweaks 
        self.n_major_actuators = int(robot_name[3:4])
        assert self.n_major_actuators == 7 
        # only tested with s3 hand
        hand_type = robot_name[4:6]
        assert hand_type == 's3'
        if hand_type == 's3':
            self.n_hand_actuators = 6 
        self.n_actuators = self.n_major_actuators + self.n_hand_actuators

        # only position tested
        self.control_type = control_type
        self.fence = fence
        self.type = 'mujoco'
        # allow agent to control this many of the joints (starting with 0)
        self.DOF = degrees_of_freedom
        self.random_state = np.random.RandomState(seed)
        self.actuated_joint_names = self.named.data.qpos.axes.row.names
        self.sub_step_limit = 10000
        self.error_step_complete = 0.01

    def step(self):
        #TODO - keep in mind type of control we are doing
        if self.control_type == 'position':
            self.step_to_position()
        else:
            raise NotImplementedError; sys.exit()

    def step_to_position(self):
        safe_step = True
        control_angles = self.data.ctrl[:self.n_major_actuators]
        joint_extremes = _find_joint_coordinate_extremes(control_angles)
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz, self.fence)
            if hit:
                print('{} DH:{} {} will hit at {} at requested joint position - blocking action'.format(self.type, xx, joint_xyz, good_xyz))
                safe_step = False
 
        if safe_step:
            cnt = 1
            super(MujocoPhysics, self).step()
            error = np.sum((control_angles - self.named.data.qpos.copy()[:self.n_major_actuators])**2)
            while cnt < self.sub_step_limit and error > self.error_step_complete:
                super(MujocoPhysics, self).step()
                error = np.sum((control_angles - self.named.data.qpos.copy()[:self.n_major_actuators])**2)
                #print('step', cnt, error)
                cnt += 1
             
    def set_position_of_target(self, target_position, target_size):
        self.named.model.geom_size['target', 0] = target_size
        self.named.model.geom_pos['target', 'x'] = target_position[0] 
        self.named.model.geom_pos['target', 'y'] = target_position[1] 
        self.named.model.geom_pos['target', 'z'] = target_position[2] 

    def reset(self):
        super(MujocoPhysics, self).reset()

    def set_joint_angles(self, body_angles):
        # fingers are always last in xml - joint angles are for major joints to least major
        self.named.data.qpos[self.actuated_joint_names[:len(body_angles)]] = body_angles

    def get_timestep(self):
        return np.array(self.timestep())

    def get_actuator_velocity(self):
        return self.named.data.actuator_velocity.copy()

    def get_actuator_force(self):
        return self.named.data.actuator_force.copy()

    def get_joint_angles_radians(self):
        # only return last joint orientation
        return self.named.data.qpos.copy()[:self.n_actuators]

    def get_joint_coordinates(self):
        return self.named.data.geom_xpos.copy()[1:self.n_actuators+1]

    def action_spec(self):
        return mujoco.action_spec(self)


class RobotPhysics(robot.Physics):
    #TODO the joint order is different for the robot and mujoco fingers - robot has major fingers, then tips. mujoco has each finger, then its tip
    def __init__(self):
        super(RobotPhysics, self).__init__()
    
    def step(self):
        safe_step = True
        joint_angles = self.data[:self.n_major_actuators]
        joint_extremes = _find_joint_coordinate_extremes(joint_angles)
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz, self.fence)
            if hit:
                #print('{} DH:{} {} will hit at {} at requested joint position - blocking action'.format(self.type, xx, joint_extremes[xx], good_xyz))
                safe_step = False
 
        if safe_step:
            super(RobotPhysics, self).step()
 
    def set_position_of_target(self, target_position, target_size):
        self.target_position = target_position

    def set_robot_position_home(self):
        # TODO - have feedback here?
        self.robot_client.home()

    def set_joint_angles(self, body_angles):
        # only send relative rad angles from here
        self.set_control(body_angles)
        self.step()

    def get_timestep(self):
        return self.timestep()

    def get_actuator_velocity(self):
        return self.actuator_velocity

    def get_actuator_force(self):
        return self.actuator_effort

    def get_joint_angles_radians(self):
        # only return last joint orientation
        return self.actuator_position

class Jaco(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, safety_physics, random=None, start_position='home', fully_observable=True, action_penalty=True, relative_step=True, relative_rad_max=.1, fence = {'x':(-1,1),'y':(-1,1),'z':(-1.2,1.2)}, degrees_of_freedom=7, extreme_joints=[4,6,7], target_size=.05, target_type='random', fixed_target_position=[.2,.2,.5]):

        """Initialize an instance of `Jaco`.
        Args:
         random: int seed number for random seed
         start_position: key indicator for where to start the robot 'home' will start in robot home position

         fully_observable: A `bool` not yet used
         action_penalty: bool impose a penalty for actions
         relative_step: bool indicates that actions are relative to previous step. Set to True for sim2real as we need to ensure that the actions trained in dm_control can be completed within the control step as they are in the real blocking ros system.
         relative_rad_max: float indicating the maximum relative step. Tested 7dof robot with .2 control_timestep and relative position of max 0.1 rads
         fence: dict with {'x':(min,max), 'y':(min,max), 'z':(min,max)} indicating a virtual cartesian fence. We impose a penalty for extreme joints which exit the virtual fence in dm_control and impose a hard limit on the real robot.
         degrees_of_freedom: int indicating the number of joints to be controlled
         extreme_joints: list of joints (starting with joint 1) to consider for imposing fence violations in dm_control. For 7dof Jaco, this should be [4,6,7]  out of joints (1,2,3,4,5,6,7).
         target_size: float indicating the size of target in reaching tasks
         target_type: string indicating if we should calculate a 'random' or 'fixed' position for the target at reset. If fixed, will used fixed_target_position
         fixed_target_position: list indicating x,y,z center of target in cartesian space
 
            location.
          relative_step: bool whether input action should be relative to current position or absolute
          relative_rad_max: float limit radian step to min/max of this value
          random: Optional,  an integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self.safe_step = False
        self.safety_physics = safety_physics
        self.target_type = target_type
        self.fixed_target_position = self.target_position = np.array(fixed_target_position)
        self.relative_step = relative_step
        self.relative_rad_max = relative_rad_max
        self.DOF = degrees_of_freedom
        self.fence = fence
        self.use_action_penalty = bool(action_penalty)
        self.extreme_joints = np.array(extreme_joints)
        self.target_size = target_size
        self.tool_position = np.zeros(3)
        self.last_tool_position = np.zeros(3)
        # ~.13 m from tool pose to fingertip
        # seems like it odesnt really train if offset < .1, def works at .15
        # radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
        self.radii = self.target_size + .15
        self.start_position = start_position
        self.random_state = np.random.RandomState(random)
        self._fully_observable = fully_observable
        self.hit_penalty = 0.0
        # TODO are open/close opposite on robot??
        self.opened_hand_position = np.zeros(6)
        self.closed_hand_position = np.array([1.1,0.1,1.1,0.1,1.1,0.1])

        # find target min/max using fence and considering table obstacle and arm reach
        # TODO Hard limits - should be made vars
        self.target_minx = max([min(self.fence['x'])]+[-.8])
        self.target_maxx = min([max(self.fence['x'])]+[.8])
        self.target_miny = max([min(self.fence['y'])]+[-.8])
        self.target_maxy = min([max(self.fence['y'])]+[.8])
        self.target_minz = max([min(self.fence['z'])]+[0.1])
        self.target_maxz = min([max(self.fence['z'])]+[.8])
        print('Jaco received virtual fence of:', self.fence)
        print('limiting target to x:({},{}), y:({},{}), z:({},{})'.format(
                               self.target_minx, self.target_maxx,
                               self.target_miny, self.target_maxy,
                               self.target_minz, self.target_maxz))
        self.sky_joint_angles = np.array([-6.27,3.27,5.17,3.24,0.234,3.54,3.14,
                                  1.1,0.0,1.1,0.0,1.1,0.])
        self.out_joint_angles = np.array([-6.27,1,5.17,3.24,0.234,3.54,3.14,
                                  1.1,0.0,1.1,0.0,1.1,0.])
 
        ## approx loc on home on real 7dof jaco2 robot
        self.sleep_joint_angles = np.array([4.71,  # 270 deg
                                  2.61,   # 150
                                  0,      # 0
                                  .5,     # 28
                                  6.28,   # 360
                                  3.7,    # 212
                                  3.14,   # 180
                                  1.1,0.1,1.1,0.1,1.1,0.1])
        # true home on the robot has the fingers open
        self.home_joint_angles = np.array([4.92,    # 283 deg
                                  2.839,   # 162.709854126
                                  0,       # 0 
                                  .758,    # 43.43
                                  4.6366,  # 265.66
                                  4.493,   # 257.47
                                  5.0249,  # 287.9
                                  1.1,0.1,1.1,0.1,1.1,0.1])
 

        super(Jaco, self).__init__()

    def observation_spec(self, physics):
        self.after_step(physics)
        super(Jaco, self).observation_spec(physics)

    def action_spec(self, physics):
        if self.relative_step or physics.type == 'robot':
            spec = specs.BoundedArray(shape=(self.DOF,), dtype=np.float, 
                                                           minimum=np.ones(self.DOF)*-self.relative_rad_max, 
                                                           maximum=np.ones(self.DOF)*self.relative_rad_max)
            return spec
        else:
            # TODO this will only work if we are using Mujoco - add to robot
            spec = physics.action_spec()
            if spec.shape[0] == self.DOF:
                return spec
            # sometimes we only want to control a few joints
            elif spec.shape[0] > self.DOF:
                return specs.BoundedArray(shape=(self.DOF,), dtype=np.float, 
                        minimum=spec.minimum[:self.DOF], 
                        maximum=spec.maximum[:self.DOF])
            else:
                raise NotImplementedError
        
    def get_position_angles_by_name(self, position_name='home'):
        if position_name == 'home':
            angles = self.home_joint_angles
        elif position_name == 'sky':
            angles = self.sky_joint_angles
        elif position_name == 'out':
            angles = self.out_joint_angles
        elif position_name == 'random':
             angles = self.find_random_joint_angles()
        else:
            raise NotImplementedError
        return angles

    def check_for_self_collisions(self, joint_angles):
        # Check for collisions. This doesnt work!
        self.safety_physics.set_joint_angles(joint_angles)
        self.safety_physics.after_reset()
        penetrating = self.safety_physics.data.ncon > 0
        return penetrating
        #print('penetrating', penetrating) 
        #if penetrating:
            # TODO we only care if target / table are hitting
            #names = self.safety_physics.named.data.xpos.axes.row.names
            #contacts_1 = self.safety_physics.data.contact.geom1
            #contacts_2 = self.safety_physics.data.contact.geom2
            ##contact_text = ['{} collides with {}'.format(names[x1], names[x2]) for x1, x2 in zip(contacts_1, contacts_2)]
            #  
            #contact_text = []
            ##for (x1, x2) in zip(contacts_1, contacts_2):
            ##    contact_text.append('{} collides with {}'.format(names[x1], names[x2]))

            #embed()
        
    def within_safety_fence(self, joint_extremes):
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz, self.fence)
            if hit:
                return False
        return True
 
    def find_random_joint_angles(self, max_trys=10000):
        safe = False
        bounds = self.safety_physics.action_spec()
        # clip rotations to one revolution
        min_bounds = bounds.minimum.clip(-np.pi*2, np.pi*2)
        max_bounds = bounds.maximum.clip(-np.pi*2, np.pi*2)
        trys = 0
        while not safe and trys < max_trys:
            random_angles = self.random_state.uniform(min_bounds, max_bounds, len(min_bounds))
            trys+=1
            if self.is_safe_position(random_angles):
                return random_angles
        print('unable to find safe random joints after {} trys'.format(trys))
        return self.home_joint_angles

    def is_safe_position(self, angles):
        joint_extremes = _find_joint_coordinate_extremes(angles[:self.safety_physics.n_major_actuators])
        if self.within_safety_fence(joint_extremes):
            if not self.check_for_self_collisions(angles):
                 return True
        return False
            
    def make_target(self, target_type='random', max_distance=1.1):
        assert target_type in ['random', 'fixed']
        if target_type == 'random':
            # limit distance from tool pose to within 1.1 meters of the base
            # TODO how does collision with the target impact physics? 
            # We don't want it to actually collide
            distance = max_distance + 10
            while distance > max_distance: 
                tx = self.random_state.uniform(self.target_minx, self.target_maxx)
                ty = self.random_state.uniform(self.target_miny, self.target_maxy)
                tz = self.random_state.uniform(self.target_minz, self.target_maxz)
                distance = np.sqrt(tx**2 + ty**2 + tz**2)
            target_position = np.array([tx,ty,tz])
        elif target_type == 'fixed':
            target_position = self.fixed_target_position
        else:
            raise ValueError; print('unknown target_type: fixed or random are required but was set to {}'.format(target_type))
        print('**setting new target position of:{}**'.format(target_position))
        return target_position


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.set_joint_angles(self.get_position_angles_by_name(self.start_position))
        self.target_position = self.make_target(self.target_type)

        # init vars
        self.joint_angles = deepcopy(physics.get_joint_angles_radians())
        self.joint_extremes = deepcopy(_find_joint_coordinate_extremes(self.joint_angles[:self.safety_physics.n_major_actuators]))
        self.tool_position = self.joint_extremes[-1]
        self.last_tool_position = self.joint_extremes[-1]
        physics.set_position_of_target(self.target_position, self.target_size)
        super(Jaco, self).initialize_episode(physics)

    def before_step(self, action, physics):
        """
        take in relative or absolute np.array of actions of length self.DOF
        if relative mode, action will be relative to the current state
        """
        self.safe_step = True
        self.hit_penalty = 0.0
        # need action to be same shape as number of actuators
        if self.relative_step:
            # relative action prone to drift over time
            relative_action = np.clip(action, -self.relative_rad_max, self.relative_rad_max)
            use_action = relative_action+self.joint_angles[:len(action)]
        else:
            use_action = np.clip(action, self.joint_angles[:len(action)]-self.relative_rad_max, self.joint_angles[:len(action)]+self.relative_rad_max)

        if len(use_action) < physics.n_actuators:
            use_action = np.hstack((use_action, self.closed_hand_position))
        joint_extremes = _find_joint_coordinate_extremes(use_action[:physics.n_major_actuators])
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz, self.fence)
            if hit:
                #print('{} DH:{} {} will hit at {} at requested joint position - blocking action'.format(physics.type, xx, joint_extremes[xx], good_xyz))
                self.hit_penalty -= 1.0
                self.safe_step = False
        if self.check_for_self_collisions(use_action):
            self.hit_penalty -= 10.0
            self.safe_step = False
        super(Jaco, self).before_step(use_action, physics)

    def after_step(self, physics):
        self.joint_angles = deepcopy(physics.get_joint_angles_radians())
        self.joint_extremes = deepcopy(_find_joint_coordinate_extremes(self.joint_angles[:7]))
        self.last_tool_position = deepcopy(self.tool_position)
        self.tool_position = self.joint_extremes[-1]
 
    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        obs['to_target'] = self.target_position-self.tool_position
        obs['joint_angles'] = self.joint_angles
        #obs['joint_forces'] = physics.get_actuator_force()
        #obs['joint_velocity'] = physics.get_actuator_velocity()
        #obs['timestep'] = physics.get_timestep()
        ## DEBUG vars
        #obs['joint_extremes'] = self.joint_extremes
        #obs['tool_position'] = self.tool_position
        #obs['jaco_link_4'] = physics.named.data.xpos['jaco_link_4']
        #obs['jaco_link_6'] = physics.named.data.xpos['jaco_link_6']
        #obs['target_position'] = self.target_position
        return obs

    def get_distance(self, position_1, position_2):
        """Returns the signed distance bt 2 positions"""
        return np.linalg.norm(position_1-position_2)

    def get_reward(self, physics):
        """Returns a sparse reward to the agent."""
        # with policy-gradient algs, the agent shakes a lot w/o some action penalty
        # Since openai-gym utilises an action penalty on its reacher, I decided to try it here
        # I've tried a penalty on the joint-position change and on the tool pose change 
        # The tool pose action penalty resulted in better (smoother) policies with TD3
        distance = self.get_distance(self.tool_position, self.target_position)
        action_penalty = -np.square(self.last_tool_position-self.tool_position).sum()
        return rewards.tolerance(distance, (0, self.radii)) + self.hit_penalty + action_penalty
