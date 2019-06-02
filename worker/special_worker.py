from env.THOR_LOADER import *
from utils.global_episode_count import _get_train_count,_add_train_count
from utils.global_episode_count import _append_target_special_roa_dict,_get_mean_target_special_roa_dict
from utils.global_episode_count import _append_init_targets_have_been_finished,_get_length_targets_have_been_finished
from config.constant import *
from config.params import *
from worker.worker import Worker


class Spe_Worker(Worker):

    def __init__(self, name, globalAC, sess, coord, N_A, N_S,device,type='Target_Special'):
        super().__init__(name=name, globalAC=globalAC, sess=sess, coord=coord,N_A= N_A,N_S= N_S,type=type,device=device)

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
        while not self.coord.should_stop() and _get_train_count() < MAX_GLOBAL_EP:
            if _get_length_targets_have_been_finished() >= len(TARGET_ID_LIST) and WHE_STOP_SPECIAL:
                print("%s targers have all been finished!"%_get_length_targets_have_been_finished())
                break
            s, t = self.env.reset_env()
            target_id = self.env.terminal_state_id
            mean_roa,roa_list = _get_mean_target_special_roa_dict(target_id)
            if mean_roa>0.9 and WHE_STOP_SPECIAL:
                _append_init_targets_have_been_finished(target_id)
            else:
                ep_r = 0
                step_in_episode = 0
                while True:
                    self.AC.load_weight(target_id=target_id)
                    a,_ = self.AC.spe_choose_action(s, t)
                    s_, r, done, info = self.env.take_action(a)
                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_t.append(t)
                    buffer_r.append(r)

                    if step_in_episode % UPDATE_SPECIAL_ITER == 0 or done:  # update global and assign to local net
                        if done:
                            v_special = 0  # terminal
                        else:
                            v_special =self.session.run(self.AC.special_v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                        buffer_v_special = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_special = r + GAMMA * v_special
                            buffer_v_special.append(v_special)

                        buffer_v_special.reverse()

                        buffer_s, buffer_a, buffer_t = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_t)
                        buffer_v_special = np.vstack(buffer_v_special)

                        feed_dict = {
                            self.AC.s: buffer_s,
                            self.AC.a: buffer_a,
                            self.AC.special_v_target: buffer_v_special,
                            }
                        self.AC.update_special(feed_dict,target_id)
                        buffer_s, buffer_a, buffer_r, buffer_t = [], [], [], []
                        self.AC.pull_special(target_id=target_id)
                    s = s_
                    total_step += 1
                    step_in_episode += 1
                    if done or step_in_episode >= MAX_STEP_IN_EPISODE:
                        if done :
                            roa = round((self.env.short_dist*1.0/step_in_episode),4)
                        else:
                            roa = 0.000
                        _append_target_special_roa_dict(target_id=target_id,roa=roa)
                    break


    def evaluate(self):
        roa,reward = super().evaluate()
        return roa,reward




