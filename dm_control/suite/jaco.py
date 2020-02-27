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
_DEFAULT_TIME_LIMIT = 30
#_DEFAULT_TIME_LIMIT = 5
#_BIG_TARGET = .05
SUITE = containers.TaggedTasks()


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
def easy(xml_name='jaco_j2s7s300.xml', time_limit=_DEFAULT_TIME_LIMIT, random_seed=None, fully_observable=True, environment_kwargs=None):
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    # hacky way to get joints
    n_joints, joint_names = get_joint_names(xml_name)
    test_target_flag=True
    physics = MujocoPhysics.from_xml_string(*get_model_and_assets(xml_name))
    task = Jaco(n_joints, joint_names, target_size=.05, test_target_flag=test_target_flag, fully_observable=fully_observable, random_seed=random_seed)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)

class MujocoPhysics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    """ NOTE when 7dof robot  is completely extended reaching for the sky in mujoco - joints are:
        [-6.27,3.27,5.17,3.24,0.234,3.54,...]
        """

    def init_robot(self, n_joints, joint_names):
        self.n_joints = n_joints
        self.joint_names = joint_names
        if self.n_joints == 7:
            # approx loc on home on real 7dof jaco2 robot
            self.home_joint_angles = [1.5722468174826083, 3.6631581056958162, 3.144652093050076,
                                 0.4417864669110646, 6.280110954017199, 3.724487776123142, 3.136988847013373,
                                 0.0012319971190548208, 0.0012319971190548208, 0.0012319971190548208, 0.0, 0.0, 0.0]


    def action_spec(self):
        """ override base class action_spec """
        return mujoco.action_spec(self)

    def set_robot_position_home(self):
        self.set_robot_position(self.joint_names, self.home_joint_angles)

    def set_robot_position(self, body_names, body_angles):
        self.named.data.qpos[body_names] = body_angles

    def bounded_joint_pos(self, joint_names):
        """Returns joint positions as (sin, cos) values."""
        joint_pos = self.named.data.qpos[joint_names]
        return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

    def get_joint_orientation(self, joint_names):
        """Returns orientations of bodies."""
        pos = self.named.data.xpos[joint_names, ['x', 'y', 'z']]
        #if orientation:
        ori = self.named.data.xquat[joint_names, ['qw', 'qx', 'qy', 'qz']]
        return np.hstack([pos, ori])

    #def get_body_geom_position(self, joint_names):
    #    """Returns position of geoms."""
    #    pos = self.named.data.geom_xpos[joint_names, ['x', 'y', 'z']]
    #    return pos

    def get_joint_position(self, joint_names):
        """Returns position of geoms."""
        pos = self.named.data.xpos[joint_names, ['x', 'y', 'z']]
        return pos


    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.named.data.qvel[joint_names]


class RobotPhysics(robot.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def action_spec(self):
        """ override base class action_spec """
        return robot.action_spec(self)



class Jaco(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, n_joints, joint_names, target_size, test_target_flag=False, fully_observable=True, random_seed=None):
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
    """
        self.n_joints = n_joints
        self._target_size = target_size
        self.random_state = np.random.RandomState(random_seed)
        self._fully_observable = fully_observable
        self.joint_names = joint_names
        #self.target_position = np.array([.5,.2,.2])
        self.target_position = np.array([.5,.5,.2])
        self.test_target_flag = test_target_flag
        super(Jaco, self).__init__()

    def action_spec(self, physics):
        return physics.action_spec()

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)

    def finger_to_target_distance(self, physics):
        """Returns the vector from target to finger in global coordinates."""
        position_finger = physics.get_joint_position(['jaco_link_finger_tip_1'])
        distance = self.get_distance(position_finger, self.target_position)
        return distance

    def get_distance(self, position_1, position_2):
        """Returns the signed distance bt 2 positions"""
        return np.linalg.norm(position_1-position_2)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # create a random target position
        physics.init_robot(self.n_joints, self.joint_names)
        # set robot home
        physics.set_robot_position_home()
        # if test flag is set, randomize target, otherwise leave it be
        if not self.test_target_flag:
            print('initializing random target')
            tx = self.random_state.uniform(-0.6, 0.6)
            ty = self.random_state.uniform(-0.6, 0.6)
            tz = self.random_state.uniform(0.2, 1.)
            self.target_position = np.array([tx,ty,tz])
        else:
            print('initializing set target')
        print(self.target_position)
        super(Jaco, self).initialize_episode(physics)

    def get_observation(self, physics):
        # """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        obs['arm_pos'] = physics.bounded_joint_pos(self.joint_names)
        obs['arm_vel'] = physics.joint_vel(self.joint_names)
        obs['to_target'] = self.finger_to_target_distance(physics)
        obs['target_pos'] = self.target_position
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        radii = self._target_size
        return rewards.tolerance(self.finger_to_target_distance(physics), (0, radii))
