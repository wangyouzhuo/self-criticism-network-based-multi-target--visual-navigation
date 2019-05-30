from env.THOR_LOADER import *
from utils.global_episode_count import _get_train_count,_add_train_count
from utils.global_episode_count import _get_steps_count,_add_steps_count
from utils.global_episode_count import _append_kl_list,_reset_kl_list,_get_kl_mean
from utils.global_episode_count import _increase_kl_beta,_decrease_kl_beta,_get_kl_beta
from utils.global_episode_count import _init_result_mean_list,_append_result_mean_list,_reset_result_mean_list,_get_train_mean_roa_reward
from utils.global_episode_count import _init_reward_roa_show,_append_reward_roa_show,_get_reward_roa_show
from config.constant import *
import numpy as np
from worker.worker import Worker


def count_list(target):
    if len(target)>0:
        count = 0
        for item in target:
            if item<0.2:
                count = count + 1
        return count*1.0/len(target)
    else:
        return 0


class Glo_Worker(Worker):

    def __init__(self, name,globalAC,sess,coord,N_A,N_S,device,type='Target_General'):
        super().__init__(name=name, globalAC=globalAC, sess=sess, coord=coord,N_A= N_A,N_S= N_S,type=type,device=device)


    def work(self):
        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
        buffer_s_next = []
        buffer_q = []
        while not self.coord.should_stop() and _get_train_count() < MAX_GLOBAL_EP:
            EPI_COUNT = _add_train_count()
            s, t = self.env.reset_env()
            kl_beta = _get_kl_beta()
            target_id = self.env.terminal_state_id
            ep_r = 0
            step_in_episode = 0
            while True:
                self.AC.load_weight(target_id=target_id)
                a,global_prob  = self.AC.glo_choose_action(s, t)

                # compute kl_divergence
                kl_dict  = {self.AC.s: np.vstack([s]),
                            self.AC.t: np.vstack([t])}
                kl = self.AC.compute_kl(kl_dict)
                _append_kl_list(kl)

                s_, r, done, info = self.env.take_action(a)

                _add_steps_count()
                ep_r += r
                buffer_s.append(s)
                buffer_s_next.append(s_)
                buffer_a.append(a)
                buffer_t.append(t)
                buffer_r.append(r)
                if _get_steps_count()%4000 == 0:
                    # compute the mean of kl : if kl>kl_max: increase kl_beta   if kl<kl_mean: decrease  kl_beta
                    kl_list,kl_mean = _get_kl_mean()
                    if kl_mean>KL_MAX:
                        _increase_kl_beta()
                    if kl_mean<KL_MIN:
                        _decrease_kl_beta()
                    kl_beta = _get_kl_beta()
                    #print("kl_beta: %6s     kl_list:%6s    kl_mean:%6s"%(round(kl_beta,4),round(count_list(kl_list),3),round(kl_mean,4)))
                    _reset_kl_list()

                if step_in_episode % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    buffer_v = self.AC.get_special_value(feed_dict={self.AC.s: buffer_s})
                    if done:
                        buffer_v[-1] = 0  # terminal
                    buffer_advantage = [0]*len(buffer_v)
                    buffer_v_next = self.AC.get_special_value(feed_dict={self.AC.s: buffer_s_next})
                    for i in range(len(buffer_r)):
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
                        self.AC.kl_beta:[kl_beta],
                        #self.AC.kl_beta:[0.0],
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
                    _append_result_mean_list(roa,ep_r)
                    if EPI_COUNT%200 == 0:
                        roa_mean_train,reward_mean_train,length_train = _get_train_mean_roa_reward()
                        _reset_result_mean_list()
                        roa_eva,reward_eva,lenght_eva = self.evaluate()
                        if length_train>100  :
                            _append_reward_roa_show(r_evaluate=reward_eva , roa_evaluate=roa_eva,
                                                    r_train=roa_mean_train, roa_train=reward_mean_train)
                            print("-------------------------------length:%s----------------------------------"%length_train)
                            print("Train!     Epi:%6s || Glo_Roa:%6s  || Glo_Reward:%7s         Evaluate!  Epi:%6s || Roa_mean:%6s || Reward_mean:%7s "
                              %(EPI_COUNT, round(roa_mean_train, 3), round(reward_mean_train, 2),EPI_COUNT,round(roa_eva,4),round(reward_eva,3)))
                    break

    def evaluate(self):
        roa,reward,length = super().evaluate()
        return roa,reward,length




