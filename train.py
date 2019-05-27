from worker.global_worker_try import *
from worker.special_worker import *
import threading
import datetime
from global_episode_count import _init_train_count,_get_result_mean_list
from global_episode_count import _init_show_list,_get_show_list,_init_roa_list
from model_try import *
from config.params import *
from config.constant import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    SESS = tf.Session()

    COORD = tf.train.Coordinator()

    N_S,N_A = 2048,4

    with tf.device(device):
        GLOBAL_AC = ACNet('Global_Net',session=SESS,N_A=N_A,N_S=N_S,device=device)  # we only need its params
        workers = []
        # Create worker
        for i in range(int(N_WORKERS * 0.5)):
            i_name = 'Spe_W_%i' % i  # worker name
            workers.append(Spe_Worker(i_name, GLOBAL_AC,sess=SESS,coord=COORD,N_A=N_A,N_S=N_S,device=device))
        for i in range(int(N_WORKERS * 0.5)):
            i_name = 'Glo_W_%i' % i  # worker name
            workers.append(Glo_Worker(i_name, GLOBAL_AC, sess=SESS, coord=COORD, N_A=N_A, N_S=N_S, device=device))

    SESS.run(tf.global_variables_initializer())

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)
    _init_train_count()
    _init_show_list()
    _init_roa_list()


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

    # roa_list,reward_list = _get_result_mean_list()
    # print(roa_list)
    # print(reward_list)
    show_list = _get_show_list()

    print('show_list',show_list)

    #text_1 = '|Alpha_incre:%s Alpha_decre:%s|%sTargets|Use_spe:%s Use_glo:%s|Fus_prob:%s Whe_adju:%s|KLmin:%s KLmax:%s|LR_A:%s LR_C:%s|LR_REG_A:%s LR_REG_C:%s' % \
    #                      (  ALPHA_INCREASE,   ALPHA_DECREASE,  len(TARGET_ID_LIST),WHE_SPECIAL_NET,WHE_NEED_GLOBAL, WHE_FUSION_PROB,WHE_ADJUST, KL_MIN, KL_MAX,LR_A,LR_C,LR_REG_A,LR_REG_C)
    plt.figure(figsize=(20, 5))
    plt.figure(1)
    plt.axis([0,len(show_list),0,1])
    plt.plot(np.arange(len(show_list)), show_list, color="r")
    plt.xlabel('hundred episodes')
    plt.ylabel('Total mean roa')
    plt.title('[ROA] :'+now_time)



    #  reward
    # plt.figure(figsize=(20, 5))
    # plt.figure(2)
    # plt.axis([0,len(reward_list),0,1])
    # plt.plot(np.arange(len(reward_list)), reward_list, color="b")
    # plt.xlabel('episodes')
    # plt.ylabel('reward')
    # plt.title('[REWARD] :'+now_time)
    #
    #
    #
    # #  kl_mean
    # plt.figure(figsize=(20, 5))
    # plt.figure(3)
    # plt.axis([0,len(GLOBAL_KL_MEAN),0,10])
    # plt.plot(np.arange(len(GLOBAL_KL_MEAN)), GLOBAL_KL_MEAN, color="g")
    # plt.xlabel('episodes')
    # plt.ylabel('KL_MEAN')
    # plt.title('[KL_MEAN] :'+now_time)
    #
    #
    plt.show()