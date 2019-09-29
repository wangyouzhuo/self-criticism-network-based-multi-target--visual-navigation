LR_A = 0.0001  # learning rate for actor

LR_SPE_A = 0.0005
LR_SPE_C = 0.001

L2_REG = 0.0001

ENTROPY_BETA = 0.1

TARGET_ID_LIST =   [
    360, 56,  265, 311, 23 ,
    66 , 245, 123, 5  , 99 ,
    333, 59,  400, 19 , 377,
    77 , 334, 257, 1  , 283,
    355, 100, 79 , 200, 2,
    4  , 232, 203, 122, 312,
    3  ,  6 ,  11, 7  ,  8 ,
    9  , 407,  88, 90 , 128,
    281, 67, 369, 405, 366,
    395, 49, 224, 188, 129,
    318, 331, 76, 48, 393,
    272, 91, 275, 404, 118
]


TARGET_ID_EVALUATE = [390, 117, 278, 50, 398, 386,
                      12, 126, 156, 231, 228, 16,
                      135, 244, 332, 254, 159, 269,
                      230, 261, 220, 85, 74,
                      15, 352, 301, 27, 190, 234,
                      250, 26, 298, 193, 63, 179,
                      361, 320, 186, 39, 314]


BETA_REG_VALUE = 0.000

BETA_REG_ACTION = 0.01

KL_MAX = 0.9

KL_MIN = 0.6

WHE_STOP_SPECIAL = False


SOFT_LOSS_TYPE = "with_soft_imitation"
# SOFT_LOSS_TYPE = "no_soft_imitation"
# SOFT_LOSS_TYPE = "hard_imitaion"