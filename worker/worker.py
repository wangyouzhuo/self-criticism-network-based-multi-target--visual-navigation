from  model_try import *
from env.THOR_LOADER import *
from global_episode_count import _get_evaluate_count,_add_evaluate_count,_init_evaluate_count,\
    _evaluate_list_mean,_reset_evaluate_count
from global_episode_count import _init_evaluate_list,_reset_evaluate_list,_append_evaluate_list,_length_evaluate_list
from global_episode_count import _init_result_mean_list,_append_result_mean_list,_reset_result_mean_list
import threading
from config.params import *
from config.constant import *
from worker.evaluater import Evaluater


class Worker(object):

    def __init__(self, name, globalAC, sess, coord, N_A, N_S, device):
        env = load_thor_env(scene_name='bedroom_04', random_start=True, random_terminal=True,
                            whe_show=False, terminal_id=None, start_id=None, whe_use_image=False,
                            whe_flatten=False, num_of_frames=1)
        self.env = env
        self.name = name
        self.AC = ACNet(scope=name, globalAC=globalAC, session=sess, N_A=N_A, N_S=N_S, device=device)
        self.session = sess
        self.coord = coord
        _init_result_mean_list()
        self.N_A = N_A
        self.N_S = N_S


    def work(self):
        pass


    def evaluate(self):
        _init_evaluate_list()
        _init_evaluate_count()
        evaluaters = []
        for i in range(N_EVALUATERS):
            evaluate_thread = threading.Thread(target=self.evaluate_thread)
            evaluaters.append(evaluate_thread)
        for thread in evaluaters:
            thread.start()
        for thread in evaluaters:
            thread.join()
        roa_mean,reward_mean = _evaluate_list_mean()
        if _length_evaluate_list()[0]>10:
            _append_result_mean_list(roa=roa_mean,reward=reward_mean)
        _reset_evaluate_list()
        _reset_evaluate_count()
        return roa_mean,reward_mean


    def evaluate_thread(self):
        evaluater = Evaluater(net=self.AC)
        s, t      = evaluater.env.reset_env()
        target_id = evaluater.env.terminal_state_id
        distance  = evaluater.env.short_dist
        evaluater.net.load_weight(target_id=target_id)
        ep_r = 0
        step_in_episode = 0
        while True:
            a, _ = evaluater.net.glo_choose_action(s, t)
            s_, r, done, info =evaluater.env.take_action(a)
            ep_r += r
            s = s_
            step_in_episode += 1
            if done or step_in_episode >= MAX_STEP_IN_EVALUATE:
                if done:
                    roa = round((distance * 1.0 / step_in_episode), 4)
                else:
                    roa = 0.0000
                _append_evaluate_list(roa=roa,reward=ep_r)
                break
        return
