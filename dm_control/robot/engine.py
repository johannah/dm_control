# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""robot`Physics` implementation and helper classes.

The `Physics` class provides the main Python interface to the robot

Robot models are defined using the MJCF XML format. The `Physics` class
can load a model from a path to an XML file, an XML string, or from a serialized
MJB binary format. See the named constructors for each of these cases.

Each `Physics` instance provides a link to the real world robot. To step forward the
simulation, use the `step` method. To set a control or actuation signal, use the
`set_control` method, which will apply the provided signal to the actuators in
subsequent calls to `step`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import collections
import contextlib
import threading

from absl import logging
#from dm_control import _render
#from dm_control.mujoco import index
#from dm_control.mujoco.wrapper import util
from dm_control.rl import control as _control
from dm_env import specs

import numpy as np
import six
import time
from IPython import embed

class RobotClient():
  def __init__(self, robot_ip="127.0.0.1", port=9030):
    self.robot_ip = robot_ip
    self.port = port
    self.connected = False
    self.startseq = '<|'
    self.endseq = '|>'
    self.midseq = '**'

  def connect(self):
    while not self.connected:
      print("attempting to connect with robot at {}".format(self.robot_ip))
      self.tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
      # connect to computer
      self.tcp_socket.connect((self.robot_ip, self.port))
      print('connected')
      self.connected = True
      if not self.connected:
        time.sleep(1)

  def send(self, cmd, msg='XX'):
    packet = self.startseq+cmd+self.midseq+msg+self.endseq
    self.tcp_socket.sendall(packet.encode())
    # TODO - should prob handle larger packets
    rx = self.tcp_socket.recv(2048).decode()
    print("rx", rx)
    return rx

  def home(self):
    return self.send('HOME')

  def reset(self):
    return self.send('RESET')

  def get_state(self):
    return self.send('GET_STATE')

  def initialize(self, minx, maxx, miny, maxy, minz, maxz):
    data = '{},{},{},{},{},{}'.format(minx,maxx, 
                                      miny,maxy, 
                                      minz,maxz)
    self.send('INIT', data)

  def step(self, command_type, relative, unit, data):
    assert(command_type in ['VEL', 'ANGLE', 'TOOL'])
    datastr = ','.join(['%.4f'%x for x in data])
    data = '{},{},{},{}'.format(command_type, relative, unit, datastr)
    return self.send('STEP', data)

  def end(self):
    self.send('END')
    print('disconnected from {}'.format(self.robot_ip))
    self.tcp_socket.close()
    self.connected = False

#class Camera(object):
#  """ scene camera.
#
#  Holds rendering properties such as the width and height of the viewport. The
#  camera position and rotation is defined by the Mujoco camera corresponding to
#  the `camera_id`. Multiple `Camera` instances may exist for a single
#  `camera_id`, for example to render the same view at different resolutions.
#  """
#
#  def __init__(self,
#               physics,
#               height=240,
#               width=320,
#               camera_id=-1,
#               max_geom=1000):
#    """Initializes a new `Camera`.
#
#    Args:
#      physics: Instance of `Physics`.
#      height: Optional image height. Defaults to 240.
#      width: Optional image width. Defaults to 320.
#      camera_id: Optional camera name or index. Defaults to -1, the free
#        camera, which is always defined. A nonnegative integer or string
#        corresponds to a fixed camera, which must be defined in the model XML.
#        If `camera_id` is a string then the camera must also be named.
#      max_geom: (optional) An integer specifying the maximum number of geoms
#        that can be represented in the scene.
#    Raises:
#      ValueError: If `camera_id` is outside the valid range, or if `width` or
#        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
#    """
#    buffer_width = physics.model.vis.global_.offwidth
#    buffer_height = physics.model.vis.global_.offheight
#    if width > buffer_width:
#      raise ValueError('Image width {} > framebuffer width {}. Either reduce '
#                       'the image width or specify a larger offscreen '
#                       'framebuffer in the model XML using the clause\n'
#                       '<visual>\n'
#                       '  <global offwidth="my_width"/>\n'
#                       '</visual>'.format(width, buffer_width))
#    if height > buffer_height:
#      raise ValueError('Image height {} > framebuffer height {}. Either reduce '
#                       'the image height or specify a larger offscreen '
#                       'framebuffer in the model XML using the clause\n'
#                       '<visual>\n'
#                       '  <global offheight="my_height"/>\n'
#                       '</visual>'.format(height, buffer_height))
#    if isinstance(camera_id, six.string_types):
#      camera_id = physics.model.name2id(camera_id, 'camera')
#    if camera_id < -1:
#      raise ValueError('camera_id cannot be smaller than -1.')
#    if camera_id >= physics.model.ncam:
#      raise ValueError('model has {} fixed cameras. camera_id={} is invalid.'.
#                       format(physics.model.ncam, camera_id))
#
#    self._width = width
#    self._height = height
#    self._physics = physics
#
#    # Variables corresponding to structs needed by Mujoco's rendering functions.
#    self._scene = wrapper.MjvScene(model=physics.model, max_geom=max_geom)
#    self._scene_option = wrapper.MjvOption()
#
#    self._perturb = wrapper.MjvPerturb()
#    self._perturb.active = 0
#    self._perturb.select = 0
#
#    self._rect = types.MJRRECT(0, 0, self._width, self._height)
#
#    self._render_camera = wrapper.MjvCamera()
#    self._render_camera.fixedcamid = camera_id
#
#    if camera_id == -1:
#      self._render_camera.type_ = enums.mjtCamera.mjCAMERA_FREE
#    else:
#      # As defined in the Mujoco documentation, mjCAMERA_FIXED refers to a
#      # camera explicitly defined in the model.
#      self._render_camera.type_ = enums.mjtCamera.mjCAMERA_FIXED
#
#    # Internal buffers.
#    self._rgb_buffer = np.empty((self._height, self._width, 3), dtype=np.uint8)
#    self._depth_buffer = np.empty((self._height, self._width), dtype=np.float32)
#
#    if self._physics.contexts.mujoco is not None:
#      with self._physics.contexts.gl.make_current() as ctx:
#        ctx.call(mjlib.mjr_setBuffer,
#                 enums.mjtFramebuffer.mjFB_OFFSCREEN,
#                 self._physics.contexts.mujoco.ptr)
#
#  @property
#  def width(self):
#    """Returns the image width (number of pixels)."""
#    return self._width
#
#  @property
#  def height(self):
#    """Returns the image height (number of pixels)."""
#    return self._height
#
#  @property
#  def option(self):
#    """Returns the camera's visualization options."""
#    return self._scene_option
#
#  @property
#  def scene(self):
#    """Returns the `mujoco.MjvScene` instance used by the camera."""
#    return self._scene
#
#  def update(self, scene_option=None):
#    """Updates geometry used for rendering.
#
#    Args:
#      scene_option: A custom `wrapper.MjvOption` instance to use to render
#        the scene instead of the default.  If None, will use the default.
#    """
#    scene_option = scene_option or self._scene_option
#    mjlib.mjv_updateScene(self._physics.model.ptr, self._physics.data.ptr,
#                          scene_option.ptr, self._perturb.ptr,
#                          self._render_camera.ptr, enums.mjtCatBit.mjCAT_ALL,
#                          self._scene.ptr)
#
#  def _render_on_gl_thread(self, depth, overlays):
#    """Performs only those rendering calls that require an OpenGL context."""
#
#    # Render the scene.
#    mjlib.mjr_render(self._rect, self._scene.ptr,
#                     self._physics.contexts.mujoco.ptr)
#
#    if not depth:
#      # If rendering RGB, draw any text overlays on top of the image.
#      for overlay in overlays:
#        overlay.draw(self._physics.contexts.mujoco.ptr, self._rect)
#
#    # Read the contents of either the RGB or depth buffer.
#    mjlib.mjr_readPixels(
#        self._rgb_buffer if not depth else None,
#        self._depth_buffer if depth else None,
#        self._rect,
#        self._physics.contexts.mujoco.ptr)
#
#  def render(self, overlays=(), depth=False, segmentation=False,
#             scene_option=None):
#    """Renders the camera view as a numpy array of pixel values.
#
#    Args:
#      overlays: An optional sequence of `TextOverlay` instances to draw. Only
#        supported if `depth` and `segmentation` are both False.
#      depth: An optional boolean. If True, makes the camera return depth
#        measurements. Cannot be enabled if `segmentation` is True.
#      segmentation: An optional boolean. If True, make the camera return a
#        pixel-wise segmentation of the scene. Cannot be enabled if `depth` is
#        True.
#      scene_option: A custom `wrapper.MjvOption` instance to use to render
#        the scene instead of the default.  If None, will use the default.
#
#    Returns:
#      The rendered scene.
#        * If `depth` and `segmentation` are both False (default), this is a
#          (height, width, 3) uint8 numpy array containing RGB values.
#        * If `depth` is True, this is a (height, width) float32 numpy array
#          containing depth values (in meters).
#        * If `segmentation` is True, this is a (height, width, 2) int32 numpy
#          array where the first channel contains the integer ID of the object at
#          each pixel, and the second channel contains the corresponding object
#          type (a value in the `mjtObj` enum). Background pixels are labeled
#          (-1, -1).
#
#    Raises:
#      ValueError: If overlays are requested with depth rendering.
#      ValueError: If both depth and segmentation flags are set together.
#    """
#
#    if overlays and (depth or segmentation):
#      raise ValueError(_OVERLAYS_NOT_SUPPORTED_FOR_DEPTH_OR_SEGMENTATION)
#
#    if depth and segmentation:
#      raise ValueError(_BOTH_SEGMENTATION_AND_DEPTH_ENABLED)
#
#    # Enable flags to compute segmentation labels
#    if segmentation:
#      self._scene.flags[enums.mjtRndFlag.mjRND_SEGMENT] = True
#      self._scene.flags[enums.mjtRndFlag.mjRND_IDCOLOR] = True
#
#    # Update scene geometry.
#    self.update(scene_option=scene_option)
#
#    # Render scene and text overlays, read contents of RGB or depth buffer.
#    with self._physics.contexts.gl.make_current() as ctx:
#      ctx.call(self._render_on_gl_thread, depth=depth, overlays=overlays)
#
#    if depth:
#      # Get the distances to the near and far clipping planes.
#      extent = self._physics.model.stat.extent
#      near = self._physics.model.vis.map_.znear * extent
#      far = self._physics.model.vis.map_.zfar * extent
#      # Convert from [0 1] to depth in meters, see links below:
#      # http://stackoverflow.com/a/6657284/1461210
#      # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
#      image = near / (1 - self._depth_buffer * (1 - near / far))
#    elif segmentation:
#      # Convert 3-channel uint8 to 1-channel uint32.
#      image3 = self._rgb_buffer.astype(np.uint32)
#      segimage = (image3[:, :, 0] +
#                  image3[:, :, 1] * (2**8) +
#                  image3[:, :, 2] * (2**16))
#      # Remap segid to 2-channel (object ID, object type) pair.
#      # Seg ID 0 is background -- will be remapped to (-1, -1).
#      segid2output = np.full((self._scene.ngeom + 1, 2), fill_value=-1,
#                             dtype=np.int32)  # Seg id cannot be > ngeom + 1.
#      visible_geoms = self._scene.geoms[self._scene.geoms.segid != -1]
#      segid2output[visible_geoms.segid + 1, 0] = visible_geoms.objid
#      segid2output[visible_geoms.segid + 1, 1] = visible_geoms.objtype
#      image = segid2output[segimage]
#    else:
#      image = self._rgb_buffer
#
#    # The first row in the buffer is the bottom row of pixels in the image.
#    return np.flipud(image)
#
#  def select(self, cursor_position):
#    """Returns bodies and geoms visible at given coordinates in the frame.
#
#    Args:
#      cursor_position:  A `tuple` containing x and y coordinates, normalized to
#        between 0 and 1, and where (0, 0) is bottom-left.
#
#    Returns:
#      A `Selected` namedtuple. Fields are None if nothing is selected.
#    """
#    self.update()
#    aspect_ratio = self._width / self._height
#    cursor_x, cursor_y = cursor_position
#    pos = np.empty(3, np.double)
#    geom_id_arr = np.intc([-1])
#    skin_id_arr = np.intc([-1])
#    body_id = mjlib.mjv_select(
#        self._physics.model.ptr,
#        self._physics.data.ptr,
#        self._scene_option.ptr,
#        aspect_ratio,
#        cursor_x,
#        cursor_y,
#        self._scene.ptr,
#        pos,
#        geom_id_arr,
#        skin_id_arr)
#    [geom_id] = geom_id_arr
#    [skin_id] = skin_id_arr
#
#    # Validate IDs
#    if body_id != -1:
#      assert 0 <= body_id < self._physics.model.nbody
#    else:
#      body_id = None
#    if geom_id != -1:
#      assert 0 <= geom_id < self._physics.model.ngeom
#    else:
#      geom_id = None
#    if skin_id != -1:
#      assert 0 <= skin_id < self._physics.model.nskin
#    else:
#      skin_id = None
#
#    if all(id_ is None for id_ in (body_id, geom_id, skin_id)):
#      pos = None
#
#    return Selected(
#        body=body_id, geom=geom_id, skin=skin_id, world_position=pos)
#
#
#class MovableCamera(Camera):
#  """Subclass of `Camera` that can be moved by changing its pose.
#
#  A `MovableCamera` always corresponds to a MuJoCo free camera with id -1.
#  """
#
#  def __init__(self, physics, height=240, width=320):
#    """Initializes a new `MovableCamera`.
#
#    Args:
#      physics: Instance of `Physics`.
#      height: Optional image height. Defaults to 240.
#      width: Optional image width. Defaults to 320.
#    """
#    super(MovableCamera, self).__init__(
#        physics=physics, height=height, width=width, camera_id=-1)
#
#  def get_pose(self):
#    """Returns the pose of the camera.
#
#    Returns:
#      A `Pose` named tuple with fields:
#        lookat: NumPy array specifying lookat point.
#        distance: Float specifying distance to `lookat`.
#        azimuth: Azimuth in degrees.
#        elevation: Elevation in degrees.
#    """
#    return Pose(self._render_camera.lookat, self._render_camera.distance,
#                self._render_camera.azimuth, self._render_camera.elevation)
#
#  def set_pose(self, lookat, distance, azimuth, elevation):
#    """Sets the pose of the camera.
#
#    Args:
#      lookat: NumPy array or list specifying lookat point.
#      distance: Float specifying distance to `lookat`.
#      azimuth: Azimuth in degrees.
#      elevation: Elevation in degrees.
#    """
#    np.copyto(self._render_camera.lookat, lookat)
#    self._render_camera.distance = distance
#    self._render_camera.azimuth = azimuth
#    self._render_camera.elevation = elevation
#
#
#class TextOverlay(object):
#  """A text overlay that can be drawn on top of a camera view."""
#
#  __slots__ = ('title', 'body', 'style', 'position')
#
#  def __init__(self, title='', body='', style='normal', position='top left'):
#    """Initializes a new TextOverlay instance.
#
#    Args:
#      title: Title text.
#      body: Body text.
#      style: The font style. Can be either "normal", "shadow", or "big".
#      position: The grid position of the overlay. Can be either "top left",
#        "top right", "bottom left", or "bottom right".
#    """
#    self.title = title
#    self.body = body
#    self.style = _FONT_STYLES[style]
#    self.position = _GRID_POSITIONS[position]
#
#  def draw(self, context, rect):
#    """Draws the overlay.
#
#    Args:
#      context: A `types.MJRCONTEXT` pointer.
#      rect: A `types.MJRRECT`.
#    """
#    mjlib.mjr_overlay(self.style,
#                      self.position,
#                      rect,
#                      util.to_binary_string(self.title),
#                      util.to_binary_string(self.body),
#                      context)
#

