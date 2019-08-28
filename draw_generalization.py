import matplotlib.pyplot as plt
import numpy as np
# draw the success rate



dict_target_20 = {320: 0.275, 193: 0.4  , 386: 0.154, 261: 0.356, 390: 1.0, 135: 0.363, 15: 0.162, 12: 0.165, 74: 0.287, 332: 0.225, 269: 0.366, 398: 1.0, 190: 0.389, 16: 0.194, 85: 0.334, 278: 0.35, 186: 0.382, 26: 0.202, 27: 0.164, 156: 0.29, 314: 0.346, 159: 0.267, 352: 0.203, 228: 0.261, 230: 0.253, 231: 0.264, 220: 0.356, 234: 0.243, 39: 0.18, 301: 0.221, 50: 0.217, 179: 0.377, 244: 0.366, 254: 0.31, 361: 0.13, 250: 0.388, 63: 0.173, 298: 0.24, 126: 0.333, 117: 0.381}
list_target_20 = []

dict_target_40 = {320: 0.282, 193: 0.411, 386: 0.168, 261: 0.379, 390: 1.0, 135: 0.343, 15: 0.161, 12: 0.169, 74: 0.275, 332: 0.204, 269: 0.372, 398: 1.0, 190: 0.391, 16: 0.185, 85: 0.331, 278: 0.361, 250: 0.385, 26: 0.21, 27: 0.173, 156: 0.278, 314: 0.346, 159: 0.274, 352: 0.19, 228: 0.237, 230: 0.242, 231: 0.233, 220: 0.339, 234: 0.253, 39: 0.175, 301: 0.205, 50: 0.222, 179: 0.405, 244: 0.36, 254: 0.319, 361: 0.144, 186: 0.384, 63: 0.188, 298: 0.213, 126: 0.321, 117: 0.367}
list_target_40 = []


dict_target_60 = {320: 0.293, 193: 0.403, 386: 0.173, 261: 0.381, 390: 1.0, 135: 0.388, 15: 0.17, 12: 0.158, 74: 0.253, 332: 0.2, 269: 0.39, 398: 1.0, 190: 0.391, 16: 0.196, 85: 0.333, 278: 0.346, 26: 0.215, 314: 0.359, 156: 0.254, 250: 0.389, 159: 0.265, 352: 0.179, 27: 0.187, 228: 0.254, 230: 0.227, 231: 0.223, 220: 0.344, 234: 0.261, 39: 0.184, 301: 0.224, 50: 0.208, 179: 0.363, 244: 0.365, 254: 0.35, 361: 0.146, 186: 0.374, 63: 0.167, 298: 0.219, 126: 0.361, 117: 0.363}
list_target_60 = []


dict_random =    {320: 0.319, 193: 0.403, 386: 0.151, 261: 0.375, 390: 1.0, 135: 0.323, 12: 0.187, 74: 0.287, 332: 0.201, 269: 0.354, 398: 1.0, 190: 0.367, 16: 0.182, 85: 0.335, 278: 0.342, 186: 0.363, 26: 0.21, 15: 0.177, 220: 0.382, 314: 0.326, 159: 0.307, 352: 0.184, 27: 0.171, 228: 0.269, 230: 0.255, 231: 0.249, 156: 0.264, 234: 0.271, 39: 0.17, 301: 0.222, 50: 0.214, 179: 0.417, 244: 0.353, 254: 0.348, 361: 0.134, 250: 0.358, 63: 0.183, 298: 0.219, 126: 0.36, 117: 0.363}
list_random = []

key_list = []

print(len(dict_target_20.keys()))

for key in dict_target_20.keys():
    # if dict_target_60[key] >= dict_target_40[key] and dict_target_40[key] >= dict_target_20[key]:
    #     if key != 361 and key != 314 and key != 190:
    key_list.append(key)


print("选出来的target_id是：",key_list)

#for key in key_list:
for key in dict_target_20.keys():
    list_target_20.append(dict_target_20[key])
    list_target_40.append(dict_target_40[key])
    list_target_60.append(dict_target_60[key])
    list_random.append(dict_random[key])


distance_list = [item for item in range(len(key_list))]
BAR_WIDTH = 0.2

x1 = distance_list
x2 = [item+BAR_WIDTH for item in distance_list]
x3 = [item+2*BAR_WIDTH for item in distance_list]
x4 = [item+3*BAR_WIDTH for item in distance_list]



# plt.xlim(0, 100000)  # 限定横轴的范围
plt.ylim(0,1.0)  # 限定纵轴的范围


plt.bar(x1,list_target_20,BAR_WIDTH,align = 'center',label="20 Targets",color = '#66B3FF')
plt.bar(x2,list_target_40,BAR_WIDTH,label="40 Targets", color='#00DB00')
plt.bar(x3,list_target_60,BAR_WIDTH,label="60 Targets", color='#FF8000')
plt.bar(x4,list_random,BAR_WIDTH,label="Random", color='#B22222')


plt.tick_params(labelsize=20)

plt.legend(prop = {'size':15})
plt.xlabel('Untrained Target ID',fontsize = 20)
plt.ylabel('Average Success Rate',fontsize = 20)
#设置x轴刻度标签
plt.xticks([item+BAR_WIDTH*0.5 for item in np.arange(len(key_list))],key_list)
plt.title('The success rate of network trained on different target quantity',fontsize = 20)
plt.show()


