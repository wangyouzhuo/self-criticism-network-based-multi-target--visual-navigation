
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
    global ROA_LIST,REWARD_LIST
    ROA_LIST,REWARD_LIST = [],[]

def _append_evaluate_list(roa,reward):
    global ROA_LIST, REWARD_LIST
    ROA_LIST.append(roa)
    REWARD_LIST.append(reward)

def _length_evaluate_list():
    global ROA_LIST, REWARD_LIST
    return len(ROA_LIST), len(REWARD_LIST)

def _evaluate_list_mean():
    global ROA_LIST, REWARD_LIST
    print("ROA_LIST : ",ROA_LIST)
    return average(ROA_LIST),average(REWARD_LIST)

def _reset_evaluate_list():
    global ROA_LIST, REWARD_LIST
    ROA_LIST, REWARD_LIST = [], []



#-----------------------------store roa_mean & reward_mean-------------------------------
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

def _get_result_mean_list():
    global ROA_MEAN, REWARD_LIST
    return average(ROA_MEAN), average(REWARD_LIST)


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


#----------------------------------------------KL_BETA------------------------------------------
def _init_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = 0.001

def _increase_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = GLOBAL_KL_BETA + 0.0005
    if GLOBAL_KL_BETA > 0.01:
        GLOBAL_KL_BETA = 0.01

def _decrease_kl_beta():
    global GLOBAL_KL_BETA
    GLOBAL_KL_BETA = GLOBAL_KL_BETA - 0.0001
    if GLOBAL_KL_BETA <= 0:
        GLOBAL_KL_BETA = 0

def _get_kl_beta():
    global GLOBAL_STEPS_COUNT
    return GLOBAL_STEPS_COUNT




def average(target):
    sum = 0
    for item in target:
        sum = sum + item
    mean = sum * 1.0 / len(target)
    return mean