from env.THOR_LOADER import *



class Evaluater(object):

    def __init__(self,net):
        self.net = net
        self.env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
                    whe_show=False, terminal_id=None, start_id=None, whe_use_image=False,whe_flatten=False, num_of_frames=1)
