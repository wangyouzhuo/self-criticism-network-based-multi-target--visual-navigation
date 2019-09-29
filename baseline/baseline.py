# -*- coding: utf-8 -*-

"""
 A3C + AI2THOR

 state：agent_current_state 1*2048

 action：0,1,2,3

 this code is a simple implement of <Target-Driven Visual Navigation In Indoor Scenes Using DRL>

"""

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.op import *
import gym
import os
import shutil
# import matplotlib.pyplot as plt
import numpy as np
from config.constant import*
from config.params import*
import tensorflow as tf
import threading
from env.THOR_LOADER import load_thor_env, get_dim
from matplotlib import pyplot as plt
from utils.op import output_record
import pandas as pd

OUTPUT_GRAPH = True
LOG_DIR = './log'
MAX_GLOBAL_EP = MAX_GLOBAL_EP
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10

LR_A = 0.00005  # learning rate for actor
LR_C = 0.00005 # learning rate for critic

GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
ENTROPY_BETA = 0.1
N_S = 2048
N_A = 4

BEST_REWARD = 0
BEST_DICT = None

device = "/gpu:0"

class ACNet(object):

    def __init__(self, scope, globalAC=None):
        tf.set_random_seed(1)
        with tf.device(device):
            if scope == GLOBAL_NET_SCOPE:  # get global network
                with tf.variable_scope(scope):
                    self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                    self.t = tf.placeholder(tf.float32, [None, N_S], 'T')

                    self.global_a_params,self.global_c_params = self._build_global_params_dict(scope)

            else:  # local net, calculate losses
                with tf.variable_scope(scope):

                    self.global_AC = globalAC

                    self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                    self.a = tf.placeholder(tf.int32, [None, ], 'A')
                    self.global_q  = tf.placeholder(tf.float32, [None, 1], 'global_Q_value')
                    self.t = tf.placeholder(tf.float32, [None, N_S], 'T')

                    self.global_a_prob,self.global_v,self.global_a_params,self.global_c_params=self._build_global_net(scope)

                    self._prepare_global_loss(scope)

                    self._prepare_global_grads(scope)

                    self._prepare_update_op(scope)

                    self._prepare_pull_op(scope)

    def _build_global_params_dict(self, scope):
        with tf.variable_scope(scope):
            # encode
            w_encode = generate_fc_weight(shape=[N_S, 512], name='global_w_encode')
            b_encode = generate_fc_bias(shape=[512]        , name='global_b_encode')
            # fusion
            w_fusion = generate_fc_weight(shape=[1024, 1024], name='global_w_f')
            b_fusion = generate_fc_bias(shape=[1024]        , name='global_b_f')
            # scene
            w_scene = generate_fc_weight(shape=[1024, 512]  , name='global_w_s')
            b_scene = generate_fc_bias(shape=[512]         , name='global_b_s')
            # actor
            w_actor = generate_fc_weight(shape=[512, N_A]    , name='global_w_a')
            b_actor = generate_fc_bias(shape=[N_A]           , name='global_b_a')
            # critic
            w_critic = generate_fc_weight(shape=[512, 1]   , name='global_w_c')
            b_critic = generate_fc_bias(shape=[1]          , name='global_b_c')
            a_params = [w_encode, b_encode, w_fusion, b_fusion,
                        w_scene, b_scene, w_actor, b_actor]
            c_params = [w_encode, b_encode, w_fusion, b_fusion,
                        w_scene, b_scene, w_critic, b_critic]
            return a_params, c_params



    def _build_global_net(self, scope):

        with tf.variable_scope(scope):
            # encode current_state&target_state --> s_encode&t_encode
            w_encode = generate_fc_weight(shape=[N_S, 512], name='w_encode')
            b_encode = generate_fc_bias(shape=[512], name='b_encode')
            s_encode = tf.nn.elu(tf.matmul(self.s, w_encode) + b_encode)
            t_encode = tf.nn.elu(tf.matmul(self.t, w_encode) + b_encode)

            concat = tf.concat([s_encode, t_encode], axis=1)

            # fusion_layer
            w_fusion = generate_fc_weight(shape=[1024, 1024], name='w_f')
            b_fusion = generate_fc_bias(shape=[1024], name='b_f')
            fusion_layer = tf.nn.elu(tf.matmul(concat, w_fusion) + b_fusion)

            # scene_layer
            w_scene = generate_fc_weight(shape=[1024, 512], name='w_s')
            b_scene = generate_fc_bias(shape=[512], name='b_s')
            scene_layer = tf.nn.elu(tf.matmul(fusion_layer, w_scene) + b_scene)

            # actor
            w_actor = generate_fc_weight(shape=[512, N_A], name='w_a')
            b_actor = generate_fc_bias(shape=[N_A], name='b_a')
            prob = tf.nn.softmax(tf.matmul(scene_layer, w_actor) + b_actor)

            # critic
            w_critic = generate_fc_weight(shape=[512, 1], name='w_c')
            b_critic = generate_fc_bias(shape=[1], name='b_c')
            value = tf.matmul(scene_layer, w_critic) + b_critic

            a_params = [w_encode, b_encode, w_fusion, b_fusion, w_scene, b_scene,w_actor, b_actor]
            c_params = [w_encode, b_encode, w_fusion, b_fusion, w_scene, b_scene,w_critic, b_critic]

            return prob, value, a_params, c_params

    def _prepare_global_loss(self,scope):
        with tf.name_scope(scope+'global_loss'):
            self.global_td = tf.subtract(self.global_q,  self.global_v, name='global_Advantage')
            with tf.name_scope('global_c_loss'):
                self.global_c_loss = tf.reduce_mean(tf.square(self.global_td))

            with tf.name_scope('global_a_loss'):
                global_log_prob = tf.reduce_sum(
                    tf.log(self.global_a_prob + 1e-5) * tf.one_hot(self.a, N_A, dtype=tf.float32), axis=1,
                    keep_dims=True)
                exp_v = global_log_prob * tf.stop_gradient(self.global_td)
                self.entropy = -tf.reduce_mean(self.global_a_prob * tf.log(self.global_a_prob + 1e-5), axis=1,keep_dims=True)  # encourage exploration
                self.exp_v = ENTROPY_BETA * self.entropy + exp_v
                self.global_a_loss = tf.reduce_mean(-self.exp_v)


    def _prepare_global_grads(self,scope):
        with tf.name_scope(scope+'global_grads'):
            with tf.name_scope('global_net_grad'):
                self.global_a_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.global_a_loss, self.global_a_params)]

                self.global_c_grads = [tf.clip_by_norm(item, 40) for item in
                                tf.gradients(self.global_c_loss, self.global_c_params)]



    def _prepare_update_op(self,scope):
        with tf.name_scope(scope+'update'):
            self.update_global_a_op = OPT_A.apply_gradients(list(zip(self.global_a_grads, self.global_AC.global_a_params)))
            self.update_global_c_op = OPT_C.apply_gradients(list(zip(self.global_c_grads, self.global_AC.global_c_params)))


    def _prepare_pull_op(self,scope):
        with tf.name_scope(scope+'pull_global_params'):
            self.pull_a_params_global = [l_p.assign(g_p) for l_p, g_p in zip(self.global_a_params, self.global_AC.global_a_params)]
            self.pull_c_params_global = [l_p.assign(g_p) for l_p, g_p in zip(self.global_c_params, self.global_AC.global_c_params)]

    def update(self, feed_dict):  # run by a local
        SESS.run([self.update_global_a_op, self.update_global_c_op], feed_dict)  # local grads applies to global net

    def pull(self):  # run by a local
        SESS.run([self.pull_a_params_global, self.pull_c_params_global])

    def choose_action(self, s, t):  # run by a local
        prob_weights = SESS.run(self.global_a_prob, feed_dict={self.s: s[np.newaxis, :],self.t: t[np.newaxis, :]} )
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action

def generate_fc_weight(shape, name):
    threshold = 1.0 / np.sqrt(shape[0])
    weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
    weight = tf.Variable(weight_matrix, name=name)
    return weight

def generate_fc_bias(shape, name):
    # bias_distribution = np.zeros(shape)
    bias_distribution = tf.constant(0.0, shape=shape)
    bias = tf.Variable(bias_distribution, name=name)
    return bias



class Worker(object):
    def __init__(self, name, globalAC):
        self.env = load_thor_env(
                                 scene_name      ='bedroom_04' , random_start = True ,
                                 random_terminal =True         , whe_show     = False,
                                 terminal_id     =None         , start_id     = None,
                                 whe_use_image   =None         , whe_flatten  = False,
                                 num_of_frames   =1
                                 )
        self.name = name
        self.AC = ACNet(scope=name, globalAC=globalAC)

    def average(self, target):
        sum = 0
        for item in target:
            sum = sum + item
        mean = sum * 1.0 / len(target)
        return mean

    def work(self):
        global GLOBAL_R, GLOBAL_EP, GLOBAL_SUM, GLOBAL_R_MEAN_LIST, GLOBAL_EP_LIST
        global GLOBAL_ROA_MEAN_LIST, GLOBAL_ROA
        global BEST_REWARD,BEST_DICT

        total_step = 1
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s, t = self.env.reset_env()
            buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
            target_id = self.env.terminal_state_id
            ep_r = 0
            step_in_episode = 0
            while True:
                a = self.AC.choose_action(s, t)
                current_id = self.env.current_state_id
                s_, r, done, info = self.env.take_action(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_t.append(t)

                if step_in_episode % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_global = 0 # terminal
                    else:
                        v_global = SESS.run(self.AC.global_v, {self.AC.s: s_[np.newaxis, :],self.AC.t: t[np.newaxis, :]})[0, 0]
                    buffer_q = []

                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_global = r + GAMMA * v_global
                        buffer_q.append(v_global)
                    buffer_q.reverse()
                    buffer_s, buffer_a, buffer_t = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_t)
                    buffer_q = np.vstack(buffer_q)
                    feed_dict = {
                        self.AC.s        : buffer_s,
                        self.AC.a        : buffer_a,
                        self.AC.global_q : buffer_q,
                        self.AC.t        : buffer_t,
                    }
                    self.AC.update(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
                    buffer_q = []
                    self.AC.pull()
                s = s_
                total_step += 1
                step_in_episode += 1
                if done or step_in_episode >= MAX_STEP_IN_EPISODE:
                    if GLOBAL_EP not in GLOBAL_EP_LIST:
                        if done:
                            roa = (self.env.short_dist * 1.0) / step_in_episode
                        else:
                            roa = 0.000
                        GLOBAL_EP_LIST.append(GLOBAL_EP)
                        GLOBAL_R_MEAN_LIST.append(ep_r)
                        GLOBAL_ROA_MEAN_LIST.append(roa)
                        if GLOBAL_EP % 100 == 0:
                            roa_mean,reward_mean = self.average(GLOBAL_ROA_MEAN_LIST),self.average(GLOBAL_R_MEAN_LIST)
                            GLOBAL_R_MEAN_LIST,GLOBAL_ROA_MEAN_LIST = [],[]
                            GLOBAL_R.append(reward_mean)
                            GLOBAL_ROA.append(roa_mean)
                            if reward_mean>BEST_REWARD:
                                BEST_REWARD = reward_mean
                                BEST_DICT = self.experiment_one()
                            if len(GLOBAL_ROA) > 1:
                                print(
                                    "Epi:%5d" % GLOBAL_EP,
                                    "|| Success: %5s" % done,
                                    "|| Steps:%4d" % step_in_episode,
                                    "|| Start:%3d" % self.env.start_state_id,
                                    "|| End:%3d" % self.env.terminal_state_id,
                                    "|| Distance:%2d" % self.env.short_dist,
                                    "|| ROA：%1.3f" % roa,
                                    "|| env_Reward：%6s" % round(ep_r, 2),
                                    "|| mean_r: %7s" % round(reward_mean, 3),
                                    "|| mean_roa: %6s" % round(roa_mean, 4),
                                )
                    GLOBAL_EP += 1
                    break
        # evaluate

    def experiment_one(self):
        TARGET_ID = 360
        target_state = self.env.get_id_state(TARGET_ID)
        result_dict = dict()
        for CURRENT_ID in [361,363,300,302,328,
                           330,333,335,48 ,50 ,
                           53, 55, 96, 95, 101,
                           103,160,162,165,167,
                           69, 71, 72, 74, 125,
                           127,200,202,205,207,
                           261, 263, 264, 266,369,
                           371, 377, 379, 380, 382,
                           40, 42, 45, 47,0,
                           2, 9, 11, 21, 23,
                           28, 30]:
            current_state = self.env.get_id_state(CURRENT_ID)
            v = SESS.run(self.AC.global_v,
                                {self.AC.s: current_state[np.newaxis, :],
                                 self.AC.t: target_state[np.newaxis , :]
                                 }
                                )[0, 0]
            result_dict[CURRENT_ID] = [v]
        return result_dict





if __name__ == "__main__":

    GLOBAL_R = []
    GLOBAL_ROA = []
    GLOBAL_EP = 0
    GLOBAL_EP_LIST = []
    GLOBAL_R_MEAN_LIST = []
    GLOBAL_ROA_MEAN_LIST = []

    SESS = tf.Session()

    with tf.device(device):
        OPT_A = tf.train.RMSPropOptimizer(LR_A , name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C , name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)


    plt.figure(figsize=(20, 5))
    plt.figure(1)
    plt.axis([0,len(GLOBAL_ROA),0,1])
    plt.plot(np.arange(len(GLOBAL_ROA)), GLOBAL_ROA, color="r")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean roa train !')
    title = 'ROA_%stargets-baseline'%(len(TARGET_ID_LIST))
    plt.title(title)

    plt.figure(figsize=(20, 5))
    plt.figure(2)
    plt.axis([0,len(GLOBAL_R),-20,20])
    plt.plot(np.arange(len(GLOBAL_R)), GLOBAL_R, color="b")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean reward train!')
    title = 'Reward_%stargets-baseline'%(len(TARGET_ID_LIST))
    plt.title(title)
    plt.show()

    # result_list,attribute_name,target_id,file_path
    path = '/home/wyz/PycharmProjects/self-criticism-network-based-multi-target--visual-navigation/output_record/baseline/'
    roa_file_name    = path + str(len(TARGET_ID_LIST)) + "targets_roa_baseline.csv"
    reward_file_name = path + str(len(TARGET_ID_LIST)) + "targets_reward_baseline.csv"
    output_record(result_list=GLOBAL_ROA , attribute_name='mean_roa',
                  target_count=len(TARGET_ID_LIST),file_path=roa_file_name)
    output_record(result_list=GLOBAL_R   , attribute_name='mean_reward',
                  target_count=len(TARGET_ID_LIST),file_path=reward_file_name)

    # resultpath = '/home/wyz/PycharmProjects/self-criticism-network-based-multi-target--visual-navigation/output_record/target360_bad_baseline.csv'
    # pd_result = pd.DataFrame(BEST_DICT)
    # dataframe2csv(pd_result,output_path=resultpath)




