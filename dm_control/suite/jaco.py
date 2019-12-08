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

import numpy as np

_CONTROL_TIMESTEP = .01
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_ARM_JOINTS = (['jaco_joint_{}'.format(i + 1) for i in range(6)] +
               ['jaco_joint_finger_{}'.format(i + 1) for i in range(3)])
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('jaco.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, fully_observable=True, environment_kwargs=None):
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Jaco(target_size=_BIG_TARGET, fully_observable=fully_observable, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def bounded_joint_pos(self, joint_names):
        """Returns joint positions as (sin, cos) values."""
        joint_pos = self.named.data.qpos[joint_names]
        return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        return self.named.data.qvel[joint_names]

    def body_pose(self, body_names, orientation=True):
        """Returns positions and/or orientations of bodies."""
        if not isinstance(body_names, str):
            body_names = np.array(body_names).reshape(-1, 1)  # Broadcast indices.
        pos = self.named.data.xpos[body_names, ['x', 'y', 'z']]
        if orientation:
            ori = self.named.data.xquat[body_names, ['qw', 'qx', 'qy', 'qz']]
            return np.hstack([pos, ori])
        else:
            return pos

    def geom_pos(self, geom_names):
        """Returns position of geoms."""
        if not isinstance(geom_names, str):
            geom_names = np.array(geom_names).reshape(-1, 1)  # Broadcast indices.
        pos = self.named.data.geom_xpos[geom_names, ['x', 'y', 'z']]
        return pos

    def site_distance(self, site1, site2):
        site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
        return np.linalg.norm(site1_to_site2)

    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return (self.named.data.geom_xpos['target', :3] -
                self.named.data.xpos['jaco_link_hand', :3])

    def finger_to_target_dist(self):
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class Jaco(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, target_size, fully_observable=True, random=None):
        """Initialize an instance of `Jaco`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      fully_observable: A `bool`, whether the observation should contain the
        position and velocity of the object being manipulated and the target
        location.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._target_size = target_size
        self._fully_observable = fully_observable
        super(Jaco, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Local aliases
        uniform = self.random.uniform
        model = physics.named.model
        data = physics.named.data

        physics.named.model.geom_size['target', 0] = self._target_size

        # Randomise angles of arm joints
        is_limited = model.jnt_limited[_ARM_JOINTS].astype(np.bool)
        joint_range = model.jnt_range[_ARM_JOINTS]
        lower_limits = np.where(is_limited, joint_range[:, 0], -np.pi)
        upper_limits = np.where(is_limited, joint_range[:, 1], np.pi)
        angles = uniform(lower_limits, upper_limits)
        data.qpos[_ARM_JOINTS] = angles

        # Randomize target position
        physics.named.model.geom_pos['target', 'x'] = self.random.uniform(-0.6, 0.6)
        physics.named.model.geom_pos['target', 'y'] = self.random.uniform(-0.6, 0.6)
        physics.named.model.geom_pos['target', 'z'] = self.random.uniform(0.2, 1.)

        super(Jaco, self).initialize_episode(physics)

    def get_observation(self, physics):
        # """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()
        obs['arm_pos'] = physics.bounded_joint_pos(_ARM_JOINTS)
        obs['arm_vel'] = physics.joint_vel(_ARM_JOINTS)
        if self._fully_observable:
            obs['hand_pos'] = physics.body_pose('jaco_link_hand')
            obs['target_pos'] = physics.geom_pos('target')
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        radii = self._target_size
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))