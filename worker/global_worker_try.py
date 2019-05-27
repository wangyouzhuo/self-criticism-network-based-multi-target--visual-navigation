from env.THOR_LOADER import *
from global_episode_count import _get_train_count,_add_train_count
from config.constant import *
import numpy as np
from worker.worker import Worker

class Glo_Worker(Worker):

    def __init__(self, name,globalAC,sess,coord,N_A,N_S,device):
        super().__init__(name=name, globalAC=globalAC, sess=sess, coord=coord,N_A= N_A,N_S= N_S,device=device)


    def work(self):
        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
        buffer_s_next = []
        buffer_q = []
        while not self.coord.should_stop() and _get_train_count() < MAX_GLOBAL_EP:
            EPI_COUNT = _add_train_count()
            s, t = self.env.reset_env()
            target_id = self.env.terminal_state_id
            ep_r = 0
            step_in_episode = 0
            while True:
                self.AC.load_weight(target_id=target_id)
                a,global_prob  = self.AC.glo_choose_action(s, t)
                '''
                _,special_prob = self.AC.spe_choose_action(s, t)
                dict  = {
                    self.AC.s: np.vstack([s]),
                    self.AC.t: np.vstack([t]),
                    self.AC.T:1,
                }
                kl = self.AC.compute_kl(dict)
                '''
                s_, r, done, info = self.env.take_action(a)
                ep_r += r
                buffer_s.append(s)
                buffer_s_next.append(s_)
                buffer_a.append(a)
                buffer_t.append(t)
                buffer_r.append(r)
                if step_in_episode % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    buffer_advantage = [0]*len(buffer_v)
                    buffer_v = self.AC.get_special_value(feed_dict={self.AC.s: buffer_s})
                    if done:
                        buffer_v[-1] = 0  # terminal
                    buffer_v_next = self.AC.get_special_value(feed_dict={self.AC.s: buffer_s_next})
                    for i in len(buffer_r):
                        v_next = buffer_v_next[i]
                        reward = buffer_r[i]
                        q = reward + GAMMA*v_next
                        advantage = q-buffer_v[i]
                        buffer_advantage[i] = advantage

                    buffer_s, buffer_a, buffer_t = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_t)
                    buffer_advantage = np.vstack(buffer_advantage)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a: buffer_a,
                        self.AC.t: buffer_t,
                        self.AC.kl_beta:[0.0000],
                        self.AC.adv:buffer_advantage
                       }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_t,buffer_s_next = [], [], [], [],[]
                    self.AC.pull_global()
                s = s_
                step_in_episode += 1
                if done or step_in_episode >= MAX_STEP_IN_EPISODE:
                    # print("regularization!")
                    if done:
                        roa = round((self.env.short_dist * 1.0 / step_in_episode), 4)
                    else:
                        roa = 0.000
                    print("Train!     Epi:%6s || Glo_Roa:%5s  || Glo_Reward:%5s" % (EPI_COUNT, round(roa, 3), round(ep_r, 2)))
                    if EPI_COUNT>100 and EPI_COUNT % EVALUATE_ITER == 0:
                        roa_eva,reward_eva = self.evaluate()
                        print("Evaluate!  Epi:%5s || Roa_mean:%6s || Reward_mean:%7s "%(EPI_COUNT,round(roa_eva,4),round(reward_eva,3)))
                    break

    def evaluate(self):
        roa,reward = super().evaluate()
        return roa,reward




