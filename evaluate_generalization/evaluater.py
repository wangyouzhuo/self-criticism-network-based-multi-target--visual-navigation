from env.THOR_LOADER import *
from config.constant import *
from config.params import *
import threading
import tensorflow as tf
import numpy as np


result_dict = dict()
N = 500


class AC_Net(object):

    def __init__(self,weight_path):

        self.weight_path = weight_path

        self._prepare_net()

        self._load_weight(weight_path)

    def _prepare_net(self):
        self.s = tf.placeholder(tf.float32, [None, 2048], 'State')
        self.t = tf.placeholder(tf.float32, [None, 2048], 'Target')
        w_encode = generate_fc_weight(shape=[self.dim_s, 512], name='global_w_encode')
        b_encode = generate_fc_bias(shape=[512], name='global_b_encode')

        # encode current_state into s_encode
        s_encode = tf.nn.elu(tf.matmul(self.s, w_encode) + b_encode)
        # encode target_state  into t_encode
        t_encode = tf.nn.elu(tf.matmul(self.t, w_encode) + b_encode) # encode target_state  into t_encode

        # s_encode||t_encode --> concat
        concat = tf.concat([s_encode, t_encode], axis=1)  # s_encode||t_encode --> concat

        # concat --> fusion_layer
        w_fusion = generate_fc_weight(shape=[1024, 512], name='global_w_f')
        b_fusion = generate_fc_bias(shape=[512], name='global_b_f')
        fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

        # fusion_layer --> scene_layer
        w_scene = generate_fc_weight(shape=[512, 512], name='global_w_s')
        b_scene = generate_fc_bias(shape=[512], name='global_b_s')
        scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

        # scene_layer --> prob
        w_actor = generate_fc_weight(shape=[512, self.dim_a], name='global_w_a')
        b_actor = generate_fc_bias(shape=[self.dim_a], name='global_b_a')
        self.global_logits = tf.matmul(scene_layer, w_actor) + b_actor
        self.prob = tf.nn.softmax(self.global_logits)


    def choose_action(self,s,t):
        prob_weights = self.session.run(self.prob,
                feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action


    def _load_weight(self):
        pass




class generalization_evaluater(object):

    def __init__(self,target_id,network):
        self.env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=False,
                                 whe_show=True, terminal_id=target_id, start_id=True, whe_use_image=True,
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
                    a,_  = self.net.choose_action(s, t)
                    s_,_, done, _ = self.env.take_action(a)
                    step = step + 1
                    if done and step<=N:
                        count = count + 1
                    elif step>N:
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

        weight_path = ROOT_PATH + 'weight/'+'xxxxx.ckpt'
        CURRENT_TARGETS_COUNT = None

        net = AC_Net(weight_path)

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

        print(CURRENT_TARGETS_COUNT)
        print(result_dict)




