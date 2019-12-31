from worker.global_worker import *
from worker.special_worker import *
import threading
import datetime
from utils.global_episode_count import _init_result_mean_list,_append_result_mean_list,_reset_result_mean_list
from utils.global_episode_count import _init_train_count,_get_train_mean_roa_reward
from utils.global_episode_count import _init_show_list,_get_show_list,_init_roa_list
from utils.global_episode_count import _init_kl_list,_append_kl_list,_get_kl_mean
from utils.global_episode_count import _init_kl_beta
from utils.global_episode_count import _init_steps_count,_add_steps_count,_reset_steps_count
from utils.global_episode_count import _init_reward_roa_show,_append_reward_roa_show,_get_reward_roa_show
from utils.global_episode_count import _init_target_special_roa_dict
from utils.global_episode_count import _init_targets_have_been_finished
from utils.global_episode_count import _init_max_reward
from model.model import *
from config.params import *
from config.constant import *
import matplotlib.pyplot as plt
from utils.op import *



if __name__ == "__main__":

    print("*************************************************************************************************")
    print('***************************Train %s targets  ||  Whe use res-feature? %s  || Whe_many_goals:%s  ****************************'
          %(str(len(TARGET_ID_LIST)),WHE_USE_RES_FEATURE,WHE_MANY_GOALS))
    print('**************************************************************************************************')

    with tf.device(device):

        config = tf.ConfigProto(allow_soft_placement=True)

        SESS = tf.Session(config=config)

        COORD = tf.train.Coordinator()

        N_S,N_A = 2048,4

        tf.set_random_seed(-1)
        GLOBAL_AC = ACNet('Global_Net',session=SESS,N_A=N_A,N_S=N_S,type=None,device=device)  # we only need its params
        workers = []
        # Create worker
        for i in range(int(N_WORKERS * 0.5)):
            i_name = 'Spe_W_%i' % i  # worker name
            workers.append(Spe_Worker(i_name, GLOBAL_AC,sess=SESS,coord=COORD,N_A=N_A,N_S=N_S,device=device))
        for i in range(int(N_WORKERS * 0.5)):
            i_name = 'Glo_W_%i' % i  # worker name
            workers.append(Glo_Worker(i_name, GLOBAL_AC, sess=SESS, coord=COORD, N_A=N_A, N_S=N_S, device=device))

        GLOBAL_AC._prepare_store()

        SESS.run(tf.global_variables_initializer())

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)
    _init_train_count()
    _init_show_list()
    _init_roa_list()
    _init_kl_list()
    _init_kl_beta()
    _init_steps_count()
    _init_result_mean_list()
    _init_reward_roa_show()
    _init_target_special_roa_dict()
    _init_targets_have_been_finished()
    _init_max_reward()



    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    #
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M ')

    #  roa
    title = now_time

    REWARD_SHOW_TRAIN,ROA_SHOW_TRAIN,_,_ = _get_reward_roa_show()

    plt.figure(figsize=(20, 5))
    plt.figure(1)
    plt.axis([0,len(ROA_SHOW_TRAIN),0,1])
    plt.plot(np.arange(len(ROA_SHOW_TRAIN)), ROA_SHOW_TRAIN, color="r")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean roa train !')
    title = 'ROA_%s_targets-my-net'%(len(TARGET_ID_LIST))
    plt.title(title+now_time)

    plt.figure(figsize=(20, 5))
    plt.figure(2)
    plt.axis([0,len(REWARD_SHOW_TRAIN),-20,20])
    plt.plot(np.arange(len(REWARD_SHOW_TRAIN)), REWARD_SHOW_TRAIN, color="b")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean reward train!')
    title = 'Reward_%s_targets-my-net'%(len(TARGET_ID_LIST))
    

    info = "My_architecture||Targets:"+str(len(TARGET_ID_LIST)) + '||Whe_use_res_feature:'\
           +str(WHE_USE_RES_FEATURE) + '||Whe_many_goals:'+str(WHE_MANY_GOALS)

    filepath = ROOT_PATH+'/output_record/experiments_about_my_architecture/'

    reward_file = info + "_Reward.csv"



    output_record(result_list=REWARD_SHOW_TRAIN,attribute_name='mean_rewards',
                  target_count=len(TARGET_ID_LIST),file_path=filepath+reward_file)

    roa_file = info + "_Roa.csv"

    output_record(result_list=ROA_SHOW_TRAIN,attribute_name='mean_roa',
              target_count=len(TARGET_ID_LIST),file_path=filepath+roa_file)

    plt.title(title+now_time)

    plt.show()