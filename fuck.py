from random import randint, sample

TARGET_ID_LIST =    [
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

target_set = set([item for item in range(408)])-set(TARGET_ID_LIST)

print(sample(list(target_set),20))