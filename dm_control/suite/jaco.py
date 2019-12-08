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
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np

_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
_P_IN_HAND = .1  # Probabillity of object-in-hand initial state
_P_IN_TARGET = .1  # Probabillity of object-in-target initial state
_ARM_JOINTS = (['jaco_joint_{}'.format(i + 1) for i in range(6)] +
               ['jaco_joint_finger_{}'.format(i + 1) for i in range(3)])
_ALL_PROPS = frozenset([])

SUITE = containers.TaggedTasks()


def make_model(use_peg, insert):
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_string = common.read_model('jaco.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)  
  return etree.tostring(mjcf, pretty_print=True), common.ASSETS

# What follows should be used as examples to create the jaco arm tasks
@SUITE.add('benchmarking', 'hard')
def bring_ball(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
               environment_kwargs=None):
  """Returns manipulator bring task with the ball prop."""
  use_peg = False
  insert = False
  physics = Physics.from_xml_string(*make_model(use_peg, insert))
  task = Bring(use_peg=use_peg, insert=insert,
               fully_observable=fully_observable, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)


@SUITE.add('hard')
def bring_peg(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns manipulator bring task with the peg prop."""
  use_peg = True
  insert = False
  physics = Physics.from_xml_string(*make_model(use_peg, insert))
  task = Bring(use_peg=use_peg, insert=insert,
               fully_observable=fully_observable, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)


@SUITE.add('hard')
def insert_ball(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
                environment_kwargs=None):
  """Returns manipulator insert task with the ball prop."""
  use_peg = False
  insert = True
  physics = Physics.from_xml_string(*make_model(use_peg, insert))
  task = Bring(use_peg=use_peg, insert=insert,
               fully_observable=fully_observable, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)


@SUITE.add('hard')
def insert_peg(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
               environment_kwargs=None):
  """Returns manipulator insert task with the peg prop."""
  use_peg = True
  insert = True
  physics = Physics.from_xml_string(*make_model(use_peg, insert))
  task = Bring(use_peg=use_peg, insert=insert,
               fully_observable=fully_observable, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""

  def bounded_joint_pos(self, joint_names):
    """Returns joint positions as (sin, cos) values."""
    joint_pos = self.named.data.qpos[joint_names]
    return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

  def joint_vel(self, joint_names):
    """Returns joint velocities."""
    return self.named.data.qvel[joint_names]

  def body_2d_pose(self, body_names, orientation=True):
    """Returns positions and/or orientations of bodies."""
    if not isinstance(body_names, str):
      body_names = np.array(body_names).reshape(-1, 1)  # Broadcast indices.
    pos = self.named.data.xpos[body_names, ['x', 'z']]
    if orientation:
      ori = self.named.data.xquat[body_names, ['qw', 'qy']]
      return np.hstack([pos, ori])
    else:
      return pos

  def touch(self):
    return np.log1p(self.data.sensordata)

  def site_distance(self, site1, site2):
    site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
    return np.linalg.norm(site1_to_site2)
  

class Bring(base.Task):
  """A Bring `Task`: bring the prop to the target."""

  def __init__(self, use_peg, insert, fully_observable, random=None):
    """Initialize an instance of the `Bring` task.

    Args:
      use_peg: A `bool`, whether to replace the ball prop with the peg prop.
      insert: A `bool`, whether to insert the prop in a receptacle.
      fully_observable: A `bool`, whether the observation should contain the
        position and velocity of the object being manipulated and the target
        location.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._fully_observable = fully_observable
    super(Bring, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # Local aliases
    choice = self.random.choice
    uniform = self.random.uniform
    model = physics.named.model
    data = physics.named.data

    # Find a collision-free random initial configuration.
    penetrating = True
    while penetrating:

      # Randomise angles of arm joints.
      is_limited = model.jnt_limited[_ARM_JOINTS].astype(np.bool)
      joint_range = model.jnt_range[_ARM_JOINTS]
      lower_limits = np.where(is_limited, joint_range[:, 0], -np.pi)
      upper_limits = np.where(is_limited, joint_range[:, 1], np.pi)
      angles = uniform(lower_limits, upper_limits)
      data.qpos[_ARM_JOINTS] = angles

      # Symmetrize hand.
      # data.qpos['finger'] = data.qpos['thumb']

      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0

    super(Bring, self).initialize_episode(physics)

  def get_observation(self, physics):
    # """Returns either features or only sensors (to be used with pixels)."""
    obs = collections.OrderedDict()
    obs['arm_pos'] = physics.bounded_joint_pos(_ARM_JOINTS)
    obs['arm_vel'] = physics.joint_vel(_ARM_JOINTS)
    obs['touch'] = physics.touch()
    if self._fully_observable:
      obs['hand_pos'] = physics.body_2d_pose('jaco_link_hand')
    # return obs

  def _is_close(self, distance):
    # return rewards.tolerance(distance, (0, _CLOSE), _CLOSE*2)
    return 0.

  def _peg_reward(self, physics):
    """Returns a reward for bringing the peg prop to the target."""
    # grasp = self._is_close(physics.site_distance('peg_grasp', 'grasp'))
    # pinch = self._is_close(physics.site_distance('peg_pinch', 'pinch'))
    # grasping = (grasp + pinch) / 2
    # bring = self._is_close(physics.site_distance('peg', 'target_peg'))
    # bring_tip = self._is_close(physics.site_distance('target_peg_tip',
    #                                                  'peg_tip'))
    # bringing = (bring + bring_tip) / 2
    # return max(bringing, grasping/3)
    return 0.

  def _ball_reward(self, physics):
    """Returns a reward for bringing the ball prop to the target."""
    # return self._is_close(physics.site_distance('ball', 'target_ball'))
    return 0.

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    # if self._use_peg:
    #   return self._peg_reward(physics)
    # else:
    #   return self._ball_reward(physics)
    return 0.