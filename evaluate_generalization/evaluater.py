from env.THOR_LOADER import *
from config.constant import *
from config.params import *
from tensorflow.python import pywrap_tensorflow
import threading
import tensorflow as tf
import numpy as np
import os


result_dict = dict()
N = 500
device = '/cpu:0'



class AC_Net(object):

    def __init__(self,weight_path,session):

        self.session = session

        self.weight_path = weight_path

        self.weight_dict = self._load_weight(weight_path)

        self._prepare_net()


    def get_weight(self,name):
        return  self.weight_dict[name]

    def _prepare_net(self):
        self.s = tf.placeholder(tf.float32, [None, 2048], 'State')
        self.t = tf.placeholder(tf.float32, [None, 2048], 'Target')
        w_encode = self.get_weight(name='global_w_encode')
        b_encode = self.get_weight(name='global_b_encode')

        # encode current_state into s_encode
        s_encode = tf.nn.elu(tf.matmul(self.s, w_encode) + b_encode)
        # encode target_state  into t_encode
        t_encode = tf.nn.elu(tf.matmul(self.t, w_encode) + b_encode) # encode target_state  into t_encode

        # s_encode||t_encode --> concat
        concat = tf.concat([s_encode, t_encode], axis=1)  # s_encode||t_encode --> concat

        # concat --> fusion_layer
        w_fusion = self.get_weight(name='global_w_f')
        b_fusion = self.get_weight(name='global_b_f')
        fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

        # fusion_layer --> scene_layer
        w_scene = self.get_weight(name='global_w_s')
        b_scene = self.get_weight(name='global_b_s')
        scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

        # scene_layer --> prob
        w_actor = self.get_weight(name='global_w_a')
        b_actor = self.get_weight(name='global_b_a')
        self.global_logits = tf.matmul(scene_layer, w_actor) + b_actor
        self.prob = tf.nn.softmax(self.global_logits)


    def choose_action(self,s,t):
        # prob_weights = self.session.run(self.prob,
        #         feed_dict={self.s: s[np.newaxis,:],self.t: t[np.newaxis, :]} )
        # action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        action = np.random.choice(range(4),p=[0.25,0.25,0.25,0.25])
        return action


    def _load_weight(self,weight_path):
        model_reader = pywrap_tensorflow.NewCheckpointReader(weight_path)
        var_dict = model_reader.get_variable_to_shape_map()
        target_dict = {'global_w_encode','global_b_encode','global_w_f','global_b_f',
                       'global_w_s','global_b_s','global_w_a','global_b_a'}
        result_dict = dict()
        for key in var_dict:
            name_list = key.split('/')
            if len(name_list)>=4 and name_list[0] == 'Global_Net' \
                    and name_list[1] == 'Global_Net' \
                    and name_list[2] in target_dict and name_list[3].split('_')[-1] == '1':
                print("variable name: ", name_list[2])
                result_dict[name_list[2]] = model_reader.get_tensor(key)
        return  result_dict




class generalization_evaluater(object):

    def __init__(self,target_id,network):
        self.env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=False,
                                 whe_show=False, terminal_id=target_id, start_id=True, whe_use_image=False,
                                 whe_flatten=False, num_of_frames=1)

        self.net = network
        self.target_id = target_id

    def evaluate(self):
        count = 0
        for i in range(1000):
            s,t = self.env.reset_env()
            target_id = self.env.terminal_state_id
            step = 0
            if target_id == self.target_id:
                while True:
                    a = self.net.choose_action(s, t)
                    s_,_, done, _ = self.env.take_action(a)
                    step = step + 1
                    if done and step<=N:
                        print("-----------target:%s--------------------current_episode:%s----------------------------steps:%s-----------"%(TARGET_ID_EVALUATE.index(self.target_id),i,step))
                        count = count + 1
                        break
                    elif step>N:
                        print("-----------target:%s--------------------current_episode:%s----------------------------Fail!-----------"%(TARGET_ID_EVALUATE.index(self.target_id),i))
                        break
            else:
                print("Error !")
                return
        success_rate = count/1000.0
        print("Evaluate_id :%s   Success_rate:%s"%(target_id,success_rate))
        result_dict[target_id] = success_rate
        return success_rate



if __name__ == '__main__':

    with tf.device(device):

        SESS = tf.Session()

        COORD = tf.train.Coordinator()

        CURRENT_TARGETS_COUNT = 60

        weight_path = ROOT_PATH + '/weight/%s_Targets.ckpt'%CURRENT_TARGETS_COUNT

        print('CURRENT EVALUATE:',weight_path)

        net = AC_Net(weight_path,session=SESS)

        evaluaters_list = []

        for id in TARGET_ID_EVALUATE:

            evaluaters_list.append(generalization_evaluater(id,net))

        evaluaters_threads = []

        for evaluater in evaluaters_list:
            job = lambda: evaluater.evaluate()
            t = threading.Thread(target=job)
            t.start()
            evaluaters_threads.append(t)

        COORD.join(evaluaters_threads)
        print('all over')
        #print(CURRENT_TARGETS_COUNT)
        print('RANDOM!!!')
        print(result_dict)




