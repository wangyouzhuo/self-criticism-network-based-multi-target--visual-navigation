from env.THOR_LOADER import *



class evaluate(object):

    def __init__(self,target_id,weight_path):

        self.env = load_thor_env()

