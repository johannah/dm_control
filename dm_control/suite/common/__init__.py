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

"""Functions to manage the common assets for domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from dm_control.utils import io as resources

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "./common/materials.xml",
    "./common/visual.xml",
    "./common/skybox.xml",
    "./common/texture_config.xml",
    #  below files are directly from kinova
    "./common/kinova_meshes/base.stl",
    "./common/kinova_meshes/shoulder.stl",
    "./common/kinova_meshes/arm.stl",
    "./common/kinova_meshes/arm_half_1.stl",
    "./common/kinova_meshes/arm_half_2.stl",
    "./common/kinova_meshes/forearm.stl",
    "./common/kinova_meshes/ring_big.stl",
    "./common/kinova_meshes/ring_small.stl",
    "./common/kinova_meshes/wrist_spherical_1.stl",
    "./common/kinova_meshes/wrist_spherical_2.stl",
    "./common/kinova_meshes/hand_3finger.stl",
    "./common/kinova_meshes/hand_3finger_main.stl",
    "./common/kinova_meshes/hand_3finger_insert.stl",
    "./common/kinova_meshes/finger_proximal.stl",
    "./common/kinova_meshes/finger_distal.stl",
    # textures
    "./common/textures/darkwood.png",
    "./common/textures/granite.png",
    "./common/textures/light_wood.png",
    "./common/textures/light_wood_v2.png",
    "./common/textures/light_wood_v3.png",
    "./common/textures/marble.png",
    "./common/textures/metal.png",
    "./common/textures/navy_blue.png",
    "./common/textures/real_door.png",
    "./common/textures/tennis.png",
    "./common/textures/wood_light.png",
    "./common/textures/wood.png",
   ]

ASSETS = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename))
          for filename in _FILENAMES}


def read_model(model_filename):
  """Reads a model XML file and returns its contents as a string."""
  return resources.GetResource(os.path.join(_SUITE_DIR, model_filename))
