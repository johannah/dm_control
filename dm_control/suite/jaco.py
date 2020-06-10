"""Jaco arm test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards

from dm_control import robot
from IPython import embed

import jaco_fence
import numpy as np

# the kinova jaco2 ros exposes the joint state at ~52Hz
#_CONTROL_TIMESTEP = .02
#_DEFAULT_TIME_LIMIT = 10
#_DEFAULT_TIME_LIMIT = 5

# size of target in meters
_BIG_TARGET = .05
_SMALL_TARGET = .015
_CLOSE_TARGET_DISTANCE = .2
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

def trim_and_check_pose_safety(position):
    """
    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
    """
    x,y,z = position
    hit = False
    if jaco_fence.maxx < x:
        x = jaco_fence.maxx
        hit = True
    if x < jaco_fence.minx:
        x = jaco_fence.minx
        hit = True
    if jaco_fence.maxy < y:
        y = jaco_fence.maxy
        hit = True
    if y < jaco_fence.miny:
        y = jaco_fence.miny
        hit = True
    if jaco_fence.maxz < z:
        z = jaco_fence.maxz
        hit = True
    if z < jaco_fence.minz:
        z = jaco_fence.minz
        hit = True
    return [x,y,z], hit

def get_model_and_assets(xml_name):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(xml_name), common.ASSETS


def extract_fence_from_xml(physics):
    fence_min_x = physics.named.model.geom_pos['low_fence_neg_x', 'x']
    fence_max_x = physics.named.model.geom_pos['low_fence_pos_x', 'x']

    fence_min_y = physics.named.model.geom_pos['low_fence_neg_y', 'y']
    fence_max_y = physics.named.model.geom_pos['low_fence_pos_y', 'y']

    fence_min_z = physics.named.model.geom_pos['low_fence_neg_x', 'z']
    fence_max_z = physics.named.model.geom_pos['high_fence_neg_x', 'z']



@SUITE.add('benchmarking', 'hard')
def reacher_hard(xml_name='jaco_j2s7s300_position.xml', random_seed=None, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and randomized target."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    #extract_fence_from_xml(physics)
    task = Jaco(target_size=_SMALL_TARGET, max_target_distance=_FAR_TARGET_DISTANCE, start_position='random', fully_observable=fully_observable, random_seed=random_seed)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, control_timestep=.02, **environment_kwargs)


@SUITE.add('benchmarking', 'medium')
def reacher_medium(xml_name='jaco_j2s7s300_position.xml', random_seed=None, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and randomized target."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    #extract_fence_from_xml(physics)
    task = Jaco(target_size=_SMALL_TARGET, max_target_distance=_FAR_TARGET_DISTANCE, start_position='home', fully_observable=fully_observable, random_seed=random_seed)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, control_timestep=.02, **environment_kwargs)

@SUITE.add('benchmarking', 'easy')
def reacher_easy(xml_name='jaco_j2s7s300_position.xml', random_seed=None, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and randomized target."""
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    #extract_fence_from_xml(physics)
    task = Jaco(target_size=_BIG_TARGET, max_target_distance=_CLOSE_TARGET_DISTANCE, start_position='home', fully_observable=fully_observable, random_seed=random_seed)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, control_timestep=.02, **environment_kwargs)

class MujocoPhysics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    """ NOTE when 7dof robot  is completely extended reaching for the sky in mujoco - joints are:
        [-6.27,3.27,5.17,3.24,0.234,3.54,...]
        """

    def initialize(self):
        # TODO - give robot name here
        # as specified in the xml files
        # find fence from xml file
        self.actuators = self.named.data.actuator_velocity.axes.row.names
        self.n_actuators = len(self.actuators)
        self.joint_names = self.named.data.qpos.axes.row.names
        # assumes that joints are ordered!
        self.actuated_joint_names = self.named.data.qpos.axes.row.names[:self.n_actuators]
        if self.n_actuators == 7:
            # approx loc on home on real 7dof jaco2 robot
            self.home_joint_angles = [4.71,  # 270 deg
                                      2.61,  # 150 
                                      0,     # 0 
                                      .5,    # 28 
                                      6.28,  # 360
                                      3.7,   # 212
                                      3.14]  # 180 
 
  
    def initialize_episode(self, start_position_function):
        start_position_function()

    def action_spec(self):
        self.initialize()
        return mujoco.action_spec(self)

    def set_robot_position_random(self):
        # TODO - should we ensure that the home position is within the fence? 
        #  we should setup walls in the xml sim
        # TODO
        self.set_robot_position(random_angles)

    def set_robot_position_home(self):
        # TODO - should we ensure that the home position is within the fence? 
        #  we should setup walls in the xml sim
        self.set_robot_position(self.home_joint_angles)

    def set_robot_position(self, body_angles):
        assert len(body_angles) == len(self.actuated_joint_names)
        self.named.data.qpos[self.actuated_joint_names] = body_angles

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
        #TODO - is finger similar enough to tool pose - prob not -- > check kinova docs
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

    def __init__(self, target_size, max_target_distance=1, start_position='home', degrees_of_freedom=7, extreme_joints=[4,6,7], fully_observable=True, random_seed=None):
        """Initialize an instance of `Jaco`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          fully_observable: A `bool`, whether the observation should contain the
            position and velocity of the object being manipulated and the target
            location.
          random: Optional,  an integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        # target + finger size
        self.radii = target_size+.01
        self.DOF = degrees_of_freedom
        self.extreme_joints = extreme_joints
        self.target_size = target_size
        self.max_target_distance = max_target_distance
        self.start_position = start_position
        self.random_state = np.random.RandomState(random_seed)
        self._fully_observable = fully_observable
        self.target_pose = [0, 0, 0]
        # need to find all joints which have actuators on them
        # Params for Denavit-Hartenberg Reference Frame Layout (DH)
        if self.DOF == 7:
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
        return physics.action_spec()

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
            physics.initialize_episode(physics.set_robot_position_home)
        else:
            physics.initialize_episode(physics.set_robot_position_random)
        physics.named.model.geom_size['target', 0] = self.target_size
        radius = self.random_state.uniform(.01, self.max_target_distance)
        theta_angle = self.random_state.uniform(0, 2*np.pi)
        phi_angle = self.random_state.uniform(0, 2*np.pi)
        # x in jaco is left/right, y is forward/back, z is up/down
        x_target_off = radius*np.sin(theta_angle)
        y_target_off = radius*np.cos(theta_angle)
        z_target_off = radius*np.sin(phi_angle)
        # determine where robot end effector is now
        x, y, z = physics.get_tool_pose()
        # ensure target is within fence
        target_pose,_ = trim_and_check_pose_safety([x+x_target_off, y+y_target_off, z+z_target_off])
        physics.named.model.geom_pos['target', 'x'] = target_pose[0] 
        physics.named.model.geom_pos['target', 'y'] = target_pose[1] 
        physics.named.model.geom_pos['target', 'z'] = target_pose[2] 
        self.target_pose = np.array(target_pose)
        #TODO - will need to use tool pose rather than finger
        super(Jaco, self).initialize_episode(physics)

    def before_step(self, action, physics):
        joint_extremes = self._find_joint_coordinate_extremes(action[:self.DOF])
        self.safe_step = True
        for xx,joint_xyz in enumerate(joint_extremes):
            good_xyz, hit = trim_and_check_pose_safety(joint_xyz)
            if hit:
                print('joint {} will hit at ({},{},{}) at requested joint position - blocking action'.format(self.extreme_joints[xx], *good_xyz))
                # the requested position is out of bounds of the fence, do not perform the action
                self.safe_step = False

        if self.safe_step:
            print(joint_xyz)
            super(Jaco, self).before_step(action, physics)
        
    def step(self, action, physics):
        if self.safe_step:
            super(Jaco, self).step(action, physics)

    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        joint_angles = physics.get_joint_angles_radians()
        # joint position starts as all zeros 
        joint_extremes = self._find_joint_coordinate_extremes(joint_angles[:self.DOF])
        obs['timestep'] = physics.get_timestep()
        obs['to_target'] = self.target_pose-joint_extremes[-1]
        obs['joint_angles'] = joint_angles 
        obs['joint_forces'] = physics.get_actuator_force()
        obs['joint_velocity'] = physics.get_actuator_velocity()
        obs['joint_extremes'] = joint_extremes
        return obs

    def get_distance(self, position_1, position_2):
        """Returns the signed distance bt 2 positions"""
        return np.linalg.norm(position_1-position_2)

    def get_reward(self, physics):
        """Returns a sparse reward to the agent."""
        distance = self.get_distance(physics.get_tool_pose(), self.target_pose)
        return rewards.tolerance(distance, (0, self.radii))
