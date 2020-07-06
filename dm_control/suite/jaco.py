"""Jaco arm test"""

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

import numpy as np

# real robot jo desk
#jaco_fence_minx = -200
#jaco_fence_maxx = 200
#jaco_fence_miny = -100
#jaco_fence_maxy = 100
#jaco_fence_minz = .15
#jaco_fence_maxz = 1.5

# the kinova jaco2 ros exposes the joint state at ~52Hz
#_DEFAULT_TIME_LIMIT = 10
_CONTROL_TIMESTEP = .02
_LONG_EPISODE_TIME_LIMIT = 20
_SHORT_EPISODE_TIME_LIMIT = 10
_TINY_EPISODE_TIME_LIMIT = 5
_BIG_TARGET = .05
_SMALL_TARGET = .015

# size of target in meters
_CLOSE_TARGET_DISTANCE = .5
_FAR_TARGET_DISTANCE = 1
SUITE = containers.TaggedTasks()

#D1 Base to shoulder 0.2755
#D2 First half upper arm length 0.2050
#D3 Second half upper arm length 0.2050
#D4 Forearm length (elbow to wrist) 0.2073
#D5 First wrist length 0.1038
#D6 Second wrist length 0.1038
#D7 Wrist to center of the hand 0.1600
#e2 Joint 3-4 lateral offset 0.0098
# TODO use xml as fence

def DHtransform(d,theta,a,alpha):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha)],
                                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])
    translation = np.array([[a*np.cos(theta)], [a*np.sin(theta)], [d]])
    last_row = np.array([[0,0,0,1]])
    T = np.vstack((np.hstack((rotation_matrix, translation)), last_row))
    return T

def trim_and_check_pose_safety(position, fence):
    """
    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
    """
    x,y,z = position
    hit = False
    safe_position = []
    for ind, dim in enumerate(['x','y','z']):
        if max(fence[dim]) < position[ind]:
            out = max(fence[dim])
            hit = True
        elif position[ind] < min(fence['x']):
            out = min(fence[dim])
            hit = True
        else:
            out = position[ind]
        safe_position.append(out)
    return safe_position, hit

def get_model_and_assets(xml_name):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(xml_name), common.ASSETS

@SUITE.add('benchmarking', 'reacher_medium')
def reacher_medium(xml_name='jaco_j2s7s300_position.xml', random=None, fully_observable=True, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, environment_kwargs={}):
    """Returns reacher with sparse reward and small/far randomized target and fixed initial robot position."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(xml_name, random)
    task = Jaco(target_size=_SMALL_TARGET, max_target_distance=_FAR_TARGET_DISTANCE, start_position='home', fence=fence, fully_observable=fully_observable, random=random)
    return control.Environment(
        physics, task, 
        control_timestep=_CONTROL_TIMESTEP, time_limit=_LONG_EPISODE_TIME_LIMIT, 
        **environment_kwargs)

@SUITE.add('benchmarking', 'relative_reacher_medium')
def relative_reacher_medium(xml_name='jaco_j2s7s300_position.xml', random=None, fully_observable=True, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, environment_kwargs={}):
    """Returns reacher with sparse reward and small/far randomized target and fixed initial robot position."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(xml_name, random)
    task = Jaco(target_size=_SMALL_TARGET, max_target_distance=_FAR_TARGET_DISTANCE, start_position='home', fence=fence, fully_observable=fully_observable, random=random)
    return control.Environment(
        physics, task, 
        control_timestep=_CONTROL_TIMESTEP, time_limit=_SHORT_EPISODE_TIME_LIMIT, 
        **environment_kwargs)

@SUITE.add('benchmarking', 'reacher_easy')
def reacher_easy(xml_name='jaco_j2s7s300_position.xml', random=None, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and large/close randomized target and fixed initial robot position."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(xml_name, random)
    task = Jaco(target_size=_BIG_TARGET, max_target_distance=_CLOSE_TARGET_DISTANCE, start_position='home', fence=fence, fully_observable=fully_observable, random=random)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, 
        control_timestep=_CONTROL_TIMESTEP, time_limit=_SHORT_EPISODE_TIME_LIMIT, 
        **environment_kwargs)

@SUITE.add('benchmarking', 'relative_reacher_easy')
def relative_reacher_easy(xml_name='jaco_j2s7s300_position.xml', random=None, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and large/close randomized target and fixed initial robot position."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(xml_name, random)
    task = Jaco(target_size=_BIG_TARGET, max_target_distance=_CLOSE_TARGET_DISTANCE, 
                start_position='home', fence=fence, fully_observable=fully_observable, random=random)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, 
        control_timestep=_CONTROL_TIMESTEP, time_limit=_SHORT_EPISODE_TIME_LIMIT, 
        **environment_kwargs)

@SUITE.add('benchmarking', 'relative_reacher_baby')
def relative_reacher_baby(xml_name='jaco_j2s7s300_position.xml', random=None, fully_observable=True, fence={'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}, target_position=[.2,-.5,.6], environment_kwargs={}):
    """Returns reacher with sparse reward and large/close fixed target and fixed initial robot position."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(xml_name, random)
    task = Jaco(target_size=_BIG_TARGET,
                start_position='home', fully_observable=fully_observable, random=random, target_type='fixed', target_position=target_position, fence=fence)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, 
        control_timestep=_CONTROL_TIMESTEP, time_limit=_TINY_EPISODE_TIME_LIMIT, 
        **environment_kwargs)

class MujocoPhysics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def initialize(self, xml_string, seed):
        self.random_state = np.random.RandomState(seed)
        self.actuated_joint_names = self.named.data.qpos.axes.row.names
        self.n_actuators = len(self.actuated_joint_names)
        self.n_major_actuators = len([n for n in self.actuated_joint_names if 'finger' not in n])
        # assumes that joints are ordered!
        if 'j2s7s300' in xml_string:
            """ NOTE when 7dof robot  is completely extended reaching for the sky in mujoco - joints are:
                [-6.27,3.27,5.17,3.24,0.234,3.54,...]
                """
            ## approx loc on home on real 7dof jaco2 robot
            self.home_joint_angles = [4.71,  # 270 deg
                                      2.61,  # 150 
                                      0,     # 0 
                                      .5,    # 28 
                                      6.28,  # 360
                                      3.7,   # 212
                                      3.14, # 180
                                      10, 10, 10, 10, 10, 10]   
            # approx loc on home on real 7dof jaco2 robot

        else:
            raise ValueError('unknown or unconfigured robot type')
 
    def set_pose_of_target(self, target_position, target_size):
        self.named.model.geom_size['target', 0] = target_size
        self.named.model.geom_pos['target', 'x'] = target_position[0] 
        self.named.model.geom_pos['target', 'y'] = target_position[1] 
        self.named.model.geom_pos['target', 'z'] = target_position[2] 

    def action_spec(self):
        return mujoco.action_spec(self)

    def set_robot_position_home(self):
        # TODO - should we ensure that the home position is within the fence? 
        #  we should setup walls in the xml sim
        self.set_robot_position(self.home_joint_angles)

    def set_robot_position(self, body_angles):
        # fingers are always last in xml - assume joint angles are for major joints to least major
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

    def state_pixels(self):
        # return camera output
        return physics.render()

    def get_tool_pose(self):
        #TODO - will need to use tool pose rather than finger
        position_finger = self.named.data.xpos['jaco_link_finger_tip_1', ['x', 'y', 'z']]
        return position_finger

class RobotPhysics():
    """Physics with additional features for the Planar Manipulator domain."""

    def __init__(self, cmd_type='vel', robot_type= 'j2n7s300', robot_server_ip='127.0.0.1', robot_server_port=9030):
        self.robot_type = robot_type
        self.robot_client = RobotClient(robot_ip=robot_server_ip, port=robot_server_port)
        #if self.robot_type == 'j2n7s300':
        #    self.n_joints = 7
        #    self.n_fingers = 3
        #    self.n_actions = int(self.n_joints + (self.n_fingers*2))
        #    self.vel_action_min = -1.0
        #    self.vel_action_max = 1.0

    #def action_spec(self):
    #    """ override base class action_spec """
    #    # TODO - this should come from the robot server rather than being hard coded here
    #    # there are different types of actiosn - this will only handle joint angle velocity commands (in radians/sec)
    #    vel_min = np.ones(self.num_actions)*self.vel_action_min
    #    vel_max = np.ones(self.num_actions)*self.vel_action_max
    #    return specs.BoundedArray(shape=(self.num_actions,), dtype=np.float, minimum=vel_min, maximum=vel_max)
    
    def joint_vel_step(joint_velocity):
        # joint_velocity is list of floats describing radians/sec of joint movement 
        step_response = self.robot_client.step('VEL', False, 'rad', joint_velocity)
        success, msg, _, n_states, time_offset, joint_ang, joint_vel, joint_effort, tool_pose = step_response

    def set_robot_position_home(self):
        return self.robot_client.home()

    def set_robot_position_random(self):
        """ TODO """

    def get_tool_pose(self):
        return self.tool_pose 

    def get_joint_angles(self, joint_names):
        """Returns position of geoms."""
        return self.joint_angles

    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.joint_velocity

class Jaco(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, target_size, max_target_distance=1, start_position='home', degrees_of_freedom=7, extreme_joints=[4,6,7], fully_observable=True, relative_step=True, relative_rad_max=.7853, random=None, target_type='random', target_position=[.2,.2,.5], 
            fence = {'x':(-1,1),'y':(-1,1),'z':(0.05,1.2)}):
        """Initialize an instance of `Jaco`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          fully_observable: A `bool`, whether the observation should contain the
            position and velocity of the object being manipulated and the target
            location.
          relative_step: bool whether input action should be relative to current position or absolute
          relative_rad_max: float limit radian step to min/max of this value
          random: Optional,  an integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self.target_type = target_type
        self.target_position = target_position
        self.relative_step = relative_step
        self.relative_rad_max = relative_rad_max
        self.DOF = degrees_of_freedom
        self.fence = fence

        self.extreme_joints = extreme_joints
        self.target_size = target_size
        # finger is ~.06  size
        self.radii = self.target_size + .1
        self.max_target_distance = max_target_distance
        self.start_position = start_position
        self.random_state = np.random.RandomState(random)
        self._fully_observable = fully_observable
        # find target min/max using fence and considering table obstacle and arm reach
        self.target_minx = max([.8*min(self.fence['x'])]+[-.75])
        self.target_maxx = min([.8*max(self.fence['x'])]+[.75])
        self.target_miny = max([.8*min(self.fence['y'])]+[-.75])
        self.target_maxy = min([.8*max(self.fence['y'])]+[.75])
        self.target_minz = max([.8*min(self.fence['z'])]+[0.01])
        self.target_maxz = min([.8*max(self.fence['z'])]+[.75])
        print('limiting target to x:({},{}), y:({},{}), z:({},{})'.format(
                               self.target_minx, self.target_maxx,
                               self.target_miny, self.target_maxy,
                               self.target_minz, self.target_maxz))

        if self.DOF in [7,13]:
            # Params for Denavit-Hartenberg Reference Frame Layout (DH)
            self.DH_lengths =  {'D1':0.2755, 'D2':0.2050, 'D3':0.2050, 'D4':0.2073, 'D5':0.1038, 'D6':0.1038, 'D7':0.1600, 'e2':0.0098}

            # DH transform from joint angle to XYZ from kinova robotics ros code
            self.DH_theta_sign = (-1, 1, 1, 1, 1, 1, 1)
            self.DH_a = (0, 0, 0, 0, 0, 0, 0)
            self.DH_d = (-self.DH_lengths['D1'], 
                         0, 
                         -(self.DH_lengths['D2']+self.DH_lengths['D3']), 
                         -self.DH_lengths['e2'], 
                         -(self.DH_lengths['D4'] + self.DH_lengths['D5']), 
                         0, 
                         -(self.DH_lengths['D6']+self.DH_lengths['D7']))
            self.DH_alpha = (np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi/2.0, np.pi)
            self.DH_theta_offset = (0, 0, 0, 0, 0, 0, 0)

        super(Jaco, self).__init__()

    def action_spec(self, physics):
        # could impose relative step size here 
        if self.relative_step:
            return specs.BoundedArray(shape=(self.DOF,), dtype=np.float, 
                                                           minimum=np.ones(self.DOF)*-self.relative_rad_max, 
                                                           maximum=np.ones(self.DOF)*self.relative_rad_max)
        else:
            spec = physics.action_spec()
            return specs.BoundedArray(shape=(self.DOF,), dtype=np.float, minimum=spec.minimum[:self.DOF], maximum=spec.maximum[:self.DOF])

    def _find_joint_coordinate_extremes(self, major_joint_angles):  
        """calculate xyz positions for joints form cartesian extremes
        major_joint_angles: ordered list of joint angles in radians (len 7 for 7DOF arm)"""
        extreme_xyz = []
        Tall = DHtransform(0.0,0.0,0.0,np.pi)
        for i, angle in enumerate(major_joint_angles):
            DH_theta = self.DH_theta_sign[i]*angle + self.DH_theta_offset[i]
            T = DHtransform(self.DH_d[i], DH_theta, self.DH_a[i], self.DH_alpha[i])
            Tall = np.dot(Tall, T)
            if i+1 in self.extreme_joints:
                # x is backwards of reality - this warrants investigation
                extreme_xyz.append([-Tall[0,3], Tall[1,3], Tall[2,3]])
        return np.array(extreme_xyz)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        if self.start_position == 'home':
            physics.set_robot_position_home()
        else:
            # TODO use tool pose to set initial position
            raise NotImplemented
        if self.target_type == 'random':
            tx = self.random_state.uniform(self.target_minx, self.target_maxx)
            ty = self.random_state.uniform(self.target_miny, self.target_maxy)
            tz = self.random_state.uniform(self.target_minz, self.target_maxz)
            self.target_position = np.array([tx,ty,tz])
        print('**setting target position of:', self.target_position)
        target_pose,_ = trim_and_check_pose_safety(self.target_position, self.fence)
        physics.set_pose_of_target(self.target_position, self.target_size)
        self.get_observation(physics)
        self.last_commanded_position = physics.get_joint_angles_radians()
        super(Jaco, self).initialize_episode(physics)

    def before_step(self, action, physics):
        if self.relative_step:
            # TODO - ensure this handles angle wraps
            use_action = np.clip(action, -self.relative_rad_max, self.relative_rad_max)
            use_action = [self.joint_angles[x]+use_action[x] for x in range(len(use_action))]
        else:
            use_action = action
        # dont requeire all joints 
        if len(use_action) < physics.n_actuators:
            use_action.extend(self.joint_angles[len(use_action):])
        self.safe_step = True
        joint_extremes = self._find_joint_coordinate_extremes(use_action[:self.DOF])
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz, self.fence)
            if hit:
                self.safe_step = False
            #    print('joint {} will hit at ({},{},{}) at requested joint position - blocking action'.format(self.extreme_joints[xx], *good_xyz))
        #        # the requested position is out of bounds of the fence, do not perform the action

        if self.safe_step:
            super(Jaco, self).before_step(use_action, physics)

    def step(self, action, physics):
        if self.safe_step:
            super(Jaco, self).step(action, physics)

    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        self.joint_angles = physics.get_joint_angles_radians()
        # joint position starts as all zeros 
        joint_extremes = self._find_joint_coordinate_extremes(self.joint_angles[:self.DOF])
        #obs['timestep'] = physics.get_timestep()
        obs['to_target'] = self.target_position-joint_extremes[-1]
        obs['joint_angles'] = self.joint_angles 
        obs['joint_forces'] = physics.get_actuator_force()
        obs['joint_velocity'] = physics.get_actuator_velocity()
        #obs['joint_extremes'] = joint_extremes
        return obs

    def get_distance(self, position_1, position_2):
        """Returns the signed distance bt 2 positions"""
        return np.linalg.norm(position_1-position_2)

    def get_reward(self, physics):
        """Returns a sparse reward to the agent."""
        distance = self.get_distance(physics.get_tool_pose(), self.target_position)
        return rewards.tolerance(distance, (0, self.radii))
