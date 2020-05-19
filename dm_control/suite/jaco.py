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

import numpy as np

# the kinova jaco2 ros exposes the joint state at ~52Hz
#_CONTROL_TIMESTEP = .02
_DEFAULT_TIME_LIMIT = 10
#_DEFAULT_TIME_LIMIT = 5
_BIG_TARGET = .05
_SMALL_TARGET = .015
SUITE = containers.TaggedTasks()

# FENCE
# left/right
minx = -.14
maxx = .25
# arm reaching toward hen facing label
miny = -.85
# arm reaching away when facing label
maxy = -.30
minz = .35
maxz = 1.5
#### default checkers - dont edit #####
assert maxx>minx
assert maxy>miny
assert maxz>minz

# TODO use xml as fence
def trim_target_pose_safety(position):
    """
    take in a position list [x,y,z] and ensure it doesn't violate the defined fence
    """
    x,y,z = position
    fence_result = ''
    if maxx < x:
        print('HIT FENCE: maxx of {} is < x of {}'.format(maxx, x))
        x = maxx
        fence_result+='+MAXFENCEX'
    if x < minx:
        print('HIT FENCE: x of {} < miny {}'.format(x, minx))
        x = minx
        fence_result+='+MINFENCEX'
    if maxy < y:
        print('HIT FENCE: maxy of {} is < y {}'.format(maxy, y))
        y = maxy
        fence_result+='+MAXFENCEY'
    if y < miny:
        print('HIT FENCE: y of {} is  miny of {}'.format(y, miny))
        y = miny
        fence_result+='MINFENCEY'
    if maxz < z:
        print('HIT FENCE: maxz of {} is < z of {}'.format(maxz, z))
        z = maxz
        fence_result+='MAXFENCEZ'
    if z < minz:
        print('HIT FENCE: z of {} < minz of {}'.format(z, minz))
        z = minz
        fence_result+='MINFENCEZ'
    return [x,y,z], fence_result


def get_joint_names(xml_name):
    if xml_name == 'jaco_j2s7s300.xml':
        n_joints = 7
    elif xml_name == 'jaco_j2s6s300.xml':
        n_joints = 6
    else:
        raise
    joint_names = (['jaco_joint_{}'.format(i + 1) for i in range(n_joints)] +
                   ['jaco_joint_finger_{}'.format(i + 1) for i in range(3)] +
                   ['jaco_joint_finger_tip_{}'.format(i + 1) for i in range(3)])
    return n_joints, joint_names

def get_model_and_assets(xml_name):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(xml_name), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(xml_name='jaco_j2s7s300.xml', random_seed=None, fully_observable=True, environment_kwargs={}):
    """Returns reacher with sparse reward and randomized target."""
    n_joints, joint_names = get_joint_names(xml_name)
    test_target_flag = True
    if 'use_robot' in environment_kwargs.keys():
        physics = RobotPhysics()
    else:
        physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
        physics.initialize(n_joints, joint_names)
    task = Jaco(n_joints, joint_names, target_size=_BIG_TARGET, test_target_flag=test_target_flag, fully_observable=fully_observable, random_seed=random_seed)
    # set n_sub_steps to repeat the action. since control_ts is at 1000 hz and real robot control ts is 50 hz, we repeat the action 20 times
    return control.Environment(
        physics, task, control_timestep=.02, **environment_kwargs)

class MujocoPhysics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    """ NOTE when 7dof robot  is completely extended reaching for the sky in mujoco - joints are:
        [-6.27,3.27,5.17,3.24,0.234,3.54,...]
        """

    def initialize(self, n_joints, joint_names):
        self.n_joints = n_joints
        self.joint_names = joint_names
        if self.n_joints == 7:
            # approx loc on home on real 7dof jaco2 robot
            self.home_joint_angles = [1.5722468174826083, 
                                      3.6631581056958162, 
                                      3.144652093050076,
                                      0.4417864669110646, 
                                      6.280110954017199, 
                                      3.724487776123142, 
                                      3.136988847013373,
                                      0.0012319971190548208, 
                                      0.0012319971190548208, 
                                      0.0012319971190548208, 0.0, 0.0, 0.0]
            """
             after home - xpos is
             In [11]: physics.named.data.xpos
                                         x         y         z
              0                  world [ 0         0         0       ]
              1            jaco_link_1 [ 0         0         0.157   ]
              2            jaco_link_2 [ 0         0.0016    0.275   ]
              3            jaco_link_3 [ 0         0.0016    0.0705  ]
              4            jaco_link_4 [ 0         0.0016   -0.135   ]
              5            jaco_link_5 [ 0        -0.0098    0.0728  ]
              6            jaco_link_6 [ 0        -0.0098    0.177   ]
              7            jaco_link_7 [ 0        -0.0098    0.0728  ]
              8     jaco_link_finger_1 [ 0.00279   0.0215   -0.0419  ]
              9 jaco_link_finger_tip_1 [ 0.0105    0.0575   -0.0661  ]
             10     jaco_link_finger_2 [ 0.13      0.0668   -0.058   ]
             11 jaco_link_finger_tip_2 [ 0.159     0.0884   -0.0332  ]
             12     jaco_link_finger_3 [ 0.109     0.0193    0.0509  ]
             13 jaco_link_finger_tip_3 [ 0.0756    0.0223    0.0799  ]

            In [12]: physics.position()
                 array([1.57224682e+00, 3.66315811e+00, 3.14465209e+00, 4.41786467e-01,
                6.28011095e+00, 3.72448778e+00, 3.13698885e+00, 1.23199712e-03,
                0.00000000e+00, 1.23199712e-03, 0.00000000e+00, 1.23199712e-03,
                0.00000000e+00])
            """
        else:
            raise


    def set_robot_position_home(self):
        # TODO - should we ensure that the home position is within the fence? 
        #  we should setup walls in the xml sim
        self.set_robot_position(self.home_joint_angles)

    def set_robot_position(self, body_angles):
        self.named.data.qpos[self.joint_names] = body_angles

    def state_pixels(self):
        # return camera output
        return physics.render()

#    def get_joint_orientation(self):
#        """Returns orientations of bodies."""
#        pos = self.named.data.xpos[self.joint_names, ['x', 'y', 'z']]
#        #if orientation:
#        ori = self.named.data.xquat[self.joint_names, ['qw', 'qx', 'qy', 'qz']]
#        return np.hstack([pos, ori])

    def get_tool_pose(self):
        #TODO - is finger similar enough to tool pose - prob not -- > check kinova docs
        position_finger = self.named.data.xpos['jaco_link_finger_tip_1', ['x', 'y', 'z']]
        return position_finger

class RobotPhysics():
    """Physics with additional features for the Planar Manipulator domain."""

    def __init__(self, cmd_type='vel', robot_type= 'j2n7s300', robot_server_ip='127.0.0.1', robot_server_port=9030):
        self.robot_type = robot_type
        self.robot_client = RobotClient(robot_ip=robot_server_ip, port=robot_server_port)
        if self.robot_type == 'j2n7s300':
            self.n_joints = 7
            self.n_fingers = 3
            self.n_actions = int(self.n_joints + (self.n_fingers*2))
            self.vel_action_min = -1.0
            self.vel_action_max = 1.0

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
        success, msg, _, n_states, time_offset, joint_pos, joint_vel, joint_effort, tool_pose = step_response

    def set_robot_position_home(self):
        return self.robot_client.home()

    def get_tool_pose(self):
        return self.tool_pose 

    def get_joint_position(self, joint_names):
        """Returns position of geoms."""
        return self.joint_position

    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.joint_velocity

class Jaco(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, n_joints, joint_names, target_size, render_cam=False, test_target_flag=False, fully_observable=True, random_seed=None,  test_x=.25, test_y=.25, test_z=.25):
        """Initialize an instance of `Jaco`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          test_target: a `bool` which when flags, will set the target to be at given location, otherwise it is random
          fully_observable: A `bool`, whether the observation should contain the
            position and velocity of the object being manipulated and the target
            location.
          random: Optional,  an integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
          test_x, test_y, test_z are points in 3D coordinate space for a non-random target. only used if test_target_flag==True
        """
        self.render_cam = render_cam
        self.n_joints = n_joints
        self._target_size = target_size
        self.random_state = np.random.RandomState(random_seed)
        self._fully_observable = fully_observable
        self.joint_names = joint_names
        self.target_pose = np.array([test_x, test_y, test_z])
        self.test_target_flag = test_target_flag
        super(Jaco, self).__init__()

    #def action_spec(self):
    #    return self.physics.action_spec()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.initialize(self.n_joints, self.joint_names)
        physics.set_robot_position_home()
        # if test flag is set, randomize target, otherwise leave it be
        # find a random offset
        physics.named.model.geom_size['target', 0] = self._target_size
        radius = self.random_state.uniform(.05, .20)
        theta_angle = self.random_state.uniform(0, 2*np.pi)
        phi_angle = self.random_state.uniform(0, 2*np.pi)
        # x in jaco is left/right, y is forward/back, z is up/down
        x_target_off = radius*np.sin(theta_angle)
        y_target_off = radius*np.cos(theta_angle)
        z_target_off = radius*np.sin(phi_angle)
        # determine where robot end effector is now
        x, y, z = physics.get_tool_pose()
        # ensure target is within fence
        target_pose, _ = trim_target_pose_safety([x+x_target_off, y+y_target_off, z+z_target_off])
        physics.named.model.geom_pos['target', 'x'] = target_pose[0] 
        physics.named.model.geom_pos['target', 'y'] = target_pose[1] 
        physics.named.model.geom_pos['target', 'z'] = target_pose[2] 
        self.target_pose = np.array(target_pose)
        #TODO - will need to use tool pose rather than finger
        self.radii = physics.named.model.geom_size[['target', 'jaco_link_finger_tip_1'], 0].sum()
        super(Jaco, self).initialize_episode(physics)

    def get_observation(self, physics):
        # """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        # joint position starts as all zeros 
        obs['position'] = physics.position()
        obs['to_target'] = self.target_pose-physics.get_tool_pose()
        obs['velocity'] = physics.velocity()
        obs['timestep'] = np.array(physics.timestep())
        return obs

    def get_distance(self, position_1, position_2):
        """Returns the signed distance bt 2 positions"""
        return np.linalg.norm(position_1-position_2)

    def get_reward(self, physics):
        """Returns a sparse reward to the agent."""
        distance = self.get_distance(physics.get_tool_pose(), self.target_pose)
        return rewards.tolerance(distance, (0, self.radii))
