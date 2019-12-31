import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from config.constant import *
from config.params import *


weight_path = ROOT_PATH + '/weight/10_Targets.ckpt'

#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(weight_path)

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()

target_dict = {'global_w_encode','global_b_encode','global_w_f','global_b_f',
               'global_w_s','global_b_s','global_w_a','global_b_a'}

result_dict = dict()

#最后，循环打印输出
for key in var_dict:
    name_list = key.split('/')
    # print(name_list)
    if len(name_list)>=4 and name_list[0] == 'Global_Net' \
            and name_list[1] == 'Global_Net' \
            and name_list[2] in target_dict and name_list[3].split('_')[-1] == '1':
        print("variable name: ", name_list[2])
        # print((model_reader.get_tensor(key)))
        result_dict[name_list[2]] = model_reader.get_tensor(key)


print(result_dict)