from random import randint, sample
from config.params import *
from config.constant import *

"""
  sample new target_id 
"""


target_set = set([item for item in range(408)])-set(TARGET_ID_LIST)

print(sample(list(target_set),40))