from env.THOR_LOADER import *
from global_episode_count import _get_train_count,_add_train_count
from global_episode_count import _append_roa_list,_get_roa_mean,_init_roa_list
from global_episode_count import _append_show_list,_init_show_list
from config.constant import *
import numpy as np
from worker.worker import Worker

class Glo_Worker(Worker):

    def __init__(self, name,globalAC,sess,coord,N_A,N_S,device):
        super().__init__(name=name, globalAC=globalAC, sess=sess, coord=coord,N_A= N_A,N_S= N_S,device=device)


    def work(self):
        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
        while not self.coord.should_stop() and _get_train_count() < MAX_GLOBAL_EP:
            EPI_COUNT = _add_train_count()
            s, t = self.env.reset_env()
            target_id = self.env.terminal_state_id
            ep_r = 0
            step_in_episode = 0
            while True:
                self.AC.load_weight(target_id=target_id)
                a,global_prob  = self.AC.glo_choose_action(s, t)
                # print(global_prob)
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
                buffer_a.append(a)
                buffer_t.append(t)
                buffer_r.append(r)
                if step_in_episode % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_global = 0  # terminal
                    else:
                        v_global = self.session.run(self.AC.global_v,
                            {self.AC.s: s_[np.newaxis, :], self.AC.t: t[np.newaxis, :]})[0, 0]
                    buffer_v_global = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_global = r + GAMMA * v_global
                        buffer_v_global.append(v_global)
                    buffer_v_global.reverse()
                    buffer_s, buffer_a, buffer_t = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_t)
                    buffer_v_global = np.vstack(buffer_v_global)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a: buffer_a,
                        self.AC.global_v_target: buffer_v_global,
                        self.AC.t: buffer_t,
                        self.AC.T: 1 }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
                    self.AC.pull_global()
                s = s_
                step_in_episode += 1
                if done or step_in_episode >= MAX_STEP_IN_EPISODE:
                    # print("regularization!")
                    if done:
                        roa = round((self.env.short_dist * 1.0 / step_in_episode), 4)
                    else:
                        roa = 0.000
                    _append_roa_list(roa)
                    # print("Train!     Epi:%6s || Glo_Roa:%5s  || Glo_Reward:%5s" % (EPI_COUNT, round(roa, 3), round(ep_r, 2)))
                    if EPI_COUNT>100 and _get_train_count() % EVALUATE_ITER == 0:
                        roa_list,roa_mean = _get_roa_mean()
                        print("EPI_COUNT: ",EPI_COUNT,"roa_len:",len(roa_list)," || roa_mean:",roa_mean,)
                        _append_show_list(roa_mean)
                        _init_roa_list()
                        # roa_eva,reward_eva = self.evaluate()
                        # print("Evaluate!  Epi:%5s || Roa_mean:%6s || Reward_mean:%7s "%(EPI_COUNT,round(roa_eva,4),round(reward_eva,3)))

                    break

    def evaluate(self):
        roa,reward = super().evaluate()
        return roa,reward




