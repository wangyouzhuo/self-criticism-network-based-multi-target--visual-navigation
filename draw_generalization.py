import matplotlib as plt

# draw the success rate



dict_target_20 = None
list_target_20 = []

dict_target_40 = None
list_target_40 = []


dict_target_60 = None
list_target_60 = []


key_list = []

for key in dict_20_targets.keys():
    if dict_target_60[key] > dict_target_40[key] and dict_target_40[key] > dict_target_20[key]:
        key_list.append()


print("选出来的target_id是：",key_list)

key_list = []
for key in key_list:
    list_target_20.append(dict_target_20[key])
    list_target_40.append(dict_target_40[key])
    list_target_60.append(dict_target_60[key])

distance_list = [item for item in range(len(distance_list))]
BAR_WIDTH = 0.4

x1 = distance_list
x2 = [item+BAR_WIDTH for item in distance_list]
x3 = [item+2*BAR_WIDTH for item in distance_list]


# plt.xlim(0, 100000)  # 限定横轴的范围
plt.ylim(0,1.0)  # 限定纵轴的范围


plt.bar(x1,list_target_20,BAR_WIDTH,align = 'center',label="20 Targets",color = '#66B3FF')
plt.bar(x2,list_target_40,BAR_WIDTH,label="40 Targets", color='#00DB00')
plt.bar(x3,list_target_60,BAR_WIDTH,label="60 Targetss", color='#FF8000')


plt.tick_params(labelsize=20)

plt.legend(prop = {'size':15})
plt.xlabel('Untrained Target ID',fontsize = 20)
plt.ylabel('Average Success Rate',fontsize = 20)
#设置x轴刻度标签
plt.xticks([item+BAR_WIDTH*0.5 for item in np.arange(len(Distance))],key_list)
plt.title('The success rate of untrained targets among network trained on different target quantity',fontsize = 20)
plt.show()


