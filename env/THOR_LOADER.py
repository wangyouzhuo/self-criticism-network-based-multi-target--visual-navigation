# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
import tensorflow as tf
from config.constant import *
from config.params import *


SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4
ACTION_SIZE = 4 # action size


# (300, 400, 3)


class THORDiscreteEnvironment(object):

  def __init__(self, config=dict()):

    # configurations
    self.scene_name          = config.get('scene_name', 'bedroom_04') # 默认使用bedroom_04房间
    self.random_start        = config.get('random_start', True)  # 随机起点
    self.random_terminal     = config.get('random_terminal',False)
    self.terminal_state_id   = config.get('terminal_state_id', 1)
    self.start_state_id      = config.get('start_state_id',1)
    self.number_of_frames    = config.get('number_of_frames',1)
    self.whe_flatten         = config.get('flatten_or_not',True)
    self.whe_use_image       = config.get('whe_use_image',False)
    self.whe_show_observation= config.get('whether_show' ,False)
    self.h5_file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
    self.h5_file      = h5py.File(self.h5_file_path, 'r')
    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.n_locations = self.locations.shape[0]

    self.transition_graph = self.h5_file['graph'][()]
    #print(self.transition_graph)
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    #self.history_length = HISTORY_LENGTH
    self.history_length = self.number_of_frames
    self.screen_height  = SCREEN_HEIGHT
    self.screen_width   = SCREEN_WIDTH

    self.N_S = 2048   # [self.frame_seq, self.screen_size[0], self.screen_size[1]]
    self.N_A = 4
    self.reset_env()

  # public methods

  def reset_env(self):

    #  choose terminal_state_id
    if self.random_terminal:
      self.terminal_state_id = random.sample(TARGET_ID_LIST,1)[0]

    #  choose start_state_id
    if self.random_start:
      random_start_id = 0
      while True:
        random_start_id = random.randrange(self.n_locations)
        dist = self.shortest_path_distances[random_start_id][self.terminal_state_id]
        if dist > 0 and random_start_id!=self.terminal_state_id:
          break
      self.start_state_id = random_start_id
    else:
      pass

    self.current_state_id = self.start_state_id
    # reset parameters
    self.short_dist = self.shortest_path_distances[self.start_state_id][self.terminal_state_id]
    if self.short_dist<0:
      raise NameError('The distance between start and terminal must large than 0 ! Please Check env_loader! ')
    self.reward   = 0
    self.collided = False
    self.terminal = False
    # self.memory   = np.zeros([M_SIZE,2048])
    self.step_count = 0
    return self.state,self.terminal_state


  def take_action(self, action):
    assert not self.terminal  , 'step() called in terminal_state'
    # 如果执行当前action不撞墙，则执行当前action
    if self.transition_graph[self.current_state_id][action] != -1:
      self.collided = False # 撞墙与否
      self.current_state_id = self.transition_graph[self.current_state_id][action].reshape([-1])[0]
      # 如果执行action后到达终点
      # print("self.terminal_state_id:%3s  ||  self.current_state_id:%3s"%(self.terminal_state_id,self.current_state_id))
      if self.terminal_state_id == self.current_state_id:
        self.terminal = True
      else:
        self.terminal = False
    # 否则 没有到终点 且 还撞墙了
    else:
      self.terminal = False
      self.collided = True
    reward = self.reward_env(self.terminal, self.collided)
    self.step_count = self.step_count + 1
    return self.state,reward,self.terminal,self.current_distance


  # s_t = s_t1
  def update(self):
    self.s_t = self.s_t1
    return

  # private methods

  def _tiled_state(self, state_id):
    f = self.h5_file['resnet_feature'][state_id][0][:,np.newaxis]
    return np.tile(f, (1, self.history_length))

  def reward_env(self, terminal, collided):
    if terminal:
        return 10.0
    if collided:
        return -0.1
    else:
        return -0.01

  # properties

  # 动作空间数量
  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE

  # 返回所有list动作名称
  @property
  def action_definitions(self):
    action_vocab = ["Forward", "Right", "Left", "Backward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def current_distance(self):
    distance = self.shortest_path_distances[self.current_state_id][self.terminal_state_id]
    return distance

  # 返回current_state_id对应的observation
  @property
  def observation(self):
    return self.h5_file['observation'][self.current_state_id]


  # 返回current_state_id对应的feature
  @property
  def state(self):
    # read from hdf5 cache
    if self.whe_use_image:
      current_state = self.observation.reshape(3, 300, 400)/255.0
      return current_state
    else:
      curent_state = self.h5_file['resnet_feature'][self.current_state_id][0][:,np.newaxis].reshape([1,-1])[0]
      return curent_state

  @property
  def terminal_state(self):
    if self.whe_use_image:
      terminal_state = self.h5_file['observation'][self.terminal_state_id].reshape(3, 300, 400)/255.0
      return terminal_state
    else:
      return self.h5_file['resnet_feature'][self.terminal_state_id][0][:,np.newaxis].reshape([1,-1])[0]



  #
  @property
  def target(self):
    return self.s_target

  #
  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]



def load_thor_env(scene_name,random_start,random_terminal,
                  whe_flatten,num_of_frames,start_id=None,
                  terminal_id=None,whe_show=False,whe_use_image=False):
    if random_start == False and start_id is None :
      raise NameError('If you want start random,please enter a Start_ID!')
      return
    if random_terminal == False and terminal_id is None:
      raise NameError('If you need random terminal,please enter a Ternimal_ID!')
      return
    if whe_flatten and whe_use_image:
      raise NameError('You can not use image and flate the state together!')
      return
    config = {
        'scene_name':scene_name,
        'random_start':random_start,
        'random_terminal': random_terminal,
        'terminal_state_id': terminal_id,
        'start_state_id':start_id,
        'number_of_frames': num_of_frames,
        'flatten_or_not':whe_flatten,
        'whe_use_image':whe_use_image,
        'whether_show':whe_show,
        'h5_file_path': ROOT_PATH + '/data/%s.h5'%scene_name,

    }
    env = THORDiscreteEnvironment(config)
    return env


def get_dim(self):
  return 4,2048



if __name__ == "__main__":
  env = load_thor_env(scene_name='bedroom_04', random_start=RANDOM_START, random_terminal=RANDOM_TERMINAL,
                      whe_show=WHE_SHOW, terminal_id=TERMINAL_ID, start_id=START_ID, whe_use_image=WHE_USE_IMAGE,
                      whe_flatten=False, num_of_frames=1)
  cur_state  = env.state
  tar_state  = env.state

  cur_state = cur_state.reshape(3, 300, 400)
  tar_state = tar_state.reshape(3, 300, 400)

  print(cur_state)
  print("--------------------------------")
  print(tar_state)


  from PIL import Image
  import numpy as n
  im = Image.fromarray(raw_state)  # numpy 转 image类
  im.show()
