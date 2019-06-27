from config.params import TARGET_ID_LIST


# ----------------------------------global count--------------------------------
def _init_train_count():
    global EPISODE_COUNT
    EPISODE_COUNT = 0

def _add_train_count():
    global EPISODE_COUNT
    EPISODE_COUNT = EPISODE_COUNT + 1
    return EPISODE_COUNT

def _get_train_count():
    global EPISODE_COUNT
    return EPISODE_COUNT

# ---------------------------------evaluate count--------------------------------
def _init_evaluate_count():
    global EVALUATE_COUNT
    EVALUATE_COUNT = 0

def _add_evaluate_count():
    global EVALUATE_COUNT
    EVALUATE_COUNT = EVALUATE_COUNT + 1

def _get_evaluate_count():
    global EVALUATE_COUNT
    return EVALUATE_COUNT

def _reset_evaluate_count():
    global EVALUATE_COUNT
    EVALUATE_COUNT = 0


# ----------------------------evaluate roa & reward list--------------------------
def _init_evaluate_list():
    global ROA_LIST_EVA,REWARD_LIST_EVA
    ROA_LIST_EVA,REWARD_LIST_EVA = [],[]

def _append_evaluate_list(roa,reward):
    global ROA_LIST_EVA, REWARD_LIST_EVA
    ROA_LIST_EVA.append(roa)
    REWARD_LIST_EVA.append(reward)

def _length_evaluate_list():
    global ROA_LIST_EVA, REWARD_LIST_EVA
    return len(ROA_LIST_EVA), len(REWARD_LIST_EVA)

def _evaluate_list_mean():
    global ROA_LIST_EVA, REWARD_LIST_EVA
    # print("ROA_LIST : ",ROA_LIST)
    length_eva = len(ROA_LIST_EVA)
    return average(ROA_LIST_EVA),average(REWARD_LIST_EVA),length_eva

def _reset_evaluate_list():
    global ROA_LIST_EVA, REWARD_LIST_EVA
    ROA_LIST_EVA, REWARD_LIST_EVA = [], []



#-----------------------------store roa_mean & reward_mean of train-------------------------------
def _init_result_mean_list():
    global ROA_MEAN,REWARD_LIST
    ROA_MEAN, REWARD_LIST = [],[]

def _append_result_mean_list(roa,reward):
    global ROA_MEAN,REWARD_LIST
    ROA_MEAN.append(roa)
    REWARD_LIST.append(reward)

def _reset_result_mean_list():
    global ROA_MEAN, REWARD_LIST
    ROA_MEAN, REWARD_LIST = [], []

def _get_train_mean_roa_reward():
    global ROA_MEAN, REWARD_LIST
    # print(REWARD_LIST)
    return average(ROA_MEAN), average(REWARD_LIST),len(REWARD_LIST)


#----------------------------------------------roa------------------------------------------
def _init_roa_list():
    global GLOBAL_ROA_LIST
    GLOBAL_ROA_LIST = []

def _append_roa_list(roa):
    global GLOBAL_ROA_LIST
    GLOBAL_ROA_LIST.append(roa)

def _get_roa_mean():
    global GLOBAL_ROA_LIST
    return GLOBAL_ROA_LIST,average(GLOBAL_ROA_LIST)


#---------------------------------------------list_to_be_show-----------------------------------
def _init_show_list():
    global GLOBAL_SHOW_LIST
    GLOBAL_SHOW_LIST = []

def _append_show_list(roa):
    global GLOBAL_SHOW_LIST
    GLOBAL_SHOW_LIST.append(roa)

def _get_show_list():
    global GLOBAL_SHOW_LIST
    return GLOBAL_SHOW_LIST


#----------------------------------------------KL_devegence------------------------------------------
def _init_kl_list():
    global GLOBAL_KL_LIST
    GLOBAL_KL_LIST = []

def _append_kl_list(kl):
    global GLOBAL_KL_LIST
    GLOBAL_KL_LIST.append(kl)

def _get_kl_mean():
    global GLOBAL_KL_LIST
    return GLOBAL_KL_LIST,average(GLOBAL_KL_LIST)

def _reset_kl_list():
    global GLOBAL_KL_LIST
    GLOBAL_KL_LIST = []


#----------------------------------------------steps_count------------------------------------------
def _init_steps_count():
    global GLOBAL_STEPS_COUNT
    GLOBAL_STEPS_COUNT = 0

def _add_steps_count():
    global GLOBAL_STEPS_COUNT
    GLOBAL_STEPS_COUNT = GLOBAL_STEPS_COUNT + 1

def _reset_steps_count():
    global GLOBAL_STEPS_COUNT
    GLOBAL_STEPS_COUNT = 0

def _get_steps_count():
    global GLOBAL_STEPS_COUNT
    return GLOBAL_STEPS_COUNT



#----------------------------------------------global_max_reward------------------------------------------
def _init_max_reward():
    global GLOBAL_MAX_REWARD
    GLOBAL_MAX_REWARD = 0

def _get_max_reward():
    global GLOBAL_MAX_REWARD
    return GLOBAL_MAX_REWARD

def _update_max_reward(max_reward):
    global GLOBAL_MAX_REWARD
    GLOBAL_MAX_REWARD = max_reward






#----------------------------------------------KL_BETA------------------------------------------
def _init_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = 0.001

def _increase_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = GLOBAL_KL_BETA + 0.0005
    if GLOBAL_KL_BETA > 0.1:
        GLOBAL_KL_BETA = 0.1

def _decrease_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = GLOBAL_KL_BETA - 0.0001
    if GLOBAL_KL_BETA <= 0:
        GLOBAL_KL_BETA = 0

def _get_kl_beta():
    global GLOBAL_KL_BETA
    return GLOBAL_KL_BETA

#----------------------------------------------REWARD_ROA_SHOW------------------------------------------

def _init_reward_roa_show():
    global REWARD_SHOW_TRAIN   , ROA_SHOW_TRAIN
    global REWARD_SHOW_EALUATE , ROA_SHOW_EVALUATE
    REWARD_SHOW_TRAIN   , ROA_SHOW_TRAIN    = [],[]
    REWARD_SHOW_EALUATE , ROA_SHOW_EVALUATE = [],[]


def _append_reward_roa_show(reward_train,roa_train,reward_evaluate,roa_evaluate):
    global REWARD_SHOW_TRAIN   , ROA_SHOW_TRAIN
    global REWARD_SHOW_EALUATE , ROA_SHOW_EVALUATE
    REWARD_SHOW_TRAIN.append(reward_train)
    ROA_SHOW_TRAIN.append(roa_train)
    REWARD_SHOW_EALUATE.append(reward_evaluate)
    ROA_SHOW_EVALUATE.append(roa_evaluate)


def _get_reward_roa_show():
    global REWARD_SHOW_TRAIN   , ROA_SHOW_TRAIN
    global REWARD_SHOW_EALUATE , ROA_SHOW_EVALUATE
    return REWARD_SHOW_TRAIN,ROA_SHOW_TRAIN,REWARD_SHOW_EALUATE,ROA_SHOW_EVALUATE

#----------------------------------------------target_special_dict--------------------------------------------
def _init_target_special_roa_dict():
    global TARGET_SPECIAL_ROA_DICT,CURRENT_TARGET_ID_LIST
    TARGET_SPECIAL_ROA_DICT = dict()
    CURRENT_TARGET_ID_LIST = set(TARGET_ID_LIST)
    for  target_id in TARGET_ID_LIST:
        TARGET_SPECIAL_ROA_DICT[target_id] = []


def _append_target_special_roa_dict(target_id,roa):
    global TARGET_SPECIAL_ROA_DICT
    TARGET_SPECIAL_ROA_DICT[target_id].append(roa)


def _get_mean_target_special_roa_dict(target_id):
    global TARGET_SPECIAL_ROA_DICT
    if len(TARGET_SPECIAL_ROA_DICT[target_id])>500:
        return average(TARGET_SPECIAL_ROA_DICT[target_id][-300:]),TARGET_SPECIAL_ROA_DICT[target_id]
    else:
        return 0,TARGET_SPECIAL_ROA_DICT[target_id]


#----------------------------------------- TARGET_ID_HAVE_BEEN_FINISHED------------------------------------

def _init_targets_have_been_finished():
    global TARGETS_HAVE_BEEN_FINISHED
    TARGETS_HAVE_BEEN_FINISHED = set()

def _append_init_targets_have_been_finished(target_id):
    global TARGETS_HAVE_BEEN_FINISHED
    if target_id in TARGETS_HAVE_BEEN_FINISHED:
        pass
    else:
        TARGETS_HAVE_BEEN_FINISHED.add(target_id)
        print("Target_id_[%s] has been finished! Special Workers have finished %s targets."%(target_id,len(TARGETS_HAVE_BEEN_FINISHED)))

def _get_length_targets_have_been_finished():
    global TARGETS_HAVE_BEEN_FINISHED
    return len(TARGETS_HAVE_BEEN_FINISHED)






def average(target):
    sum = 0
    for item in target:
        sum = sum + item
    mean = sum * 1.0 / len(target)
    return mean