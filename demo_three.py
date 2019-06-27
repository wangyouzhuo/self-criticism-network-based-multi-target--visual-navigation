import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from scipy.interpolate import spline


"""
baseline 的训练时间对比:

    excel --> 图片
"""


def clip_value(x):
    if x>= -10:
        return x
    else:
        return -10


#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

xlsx_path ='/home/wyz/PycharmProjects/self-criticism-network-based-multi-target--visual-navigation/output_record/baseline_excel.xlsx'

pd_data = pd.read_excel(xlsx_path)

print(pd_data.columns)

color_list = ['#ed1941','#ae6642','#585eaa','#00ae9d','#33a3dc','#ffd400']

x = [100*item for item in pd_data.index.tolist()]
# xnew = np.linspace(min(x),max(x),400)

y_dict = dict()
new_list = [str(item)[0:str(item).index('_')] for item in pd_data.columns.tolist()]
new_list.sort()
print(new_list)


for i in range(len(pd_data.columns)):
    quantity = new_list[i]
    column = str(quantity)+"_reward_per100_episode"
    attribute = str(quantity)+' targets'
    y = pd_data[column].tolist()
    y = [clip_value(item) for item in y]
    y_dict[attribute] = y
    plt.plot(x, y,color = color_list[i],label=attribute,linewidth=1.5)



#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(0, 100000)  # 限定横轴的范围
plt.ylim(-10,10)  # 限定纵轴的范围

plt.tick_params(labelsize=30)

plt.legend(fontsize=30)  # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Episodes",fontsize = 30) #X轴标签
plt.ylabel("Average reward per episode",fontsize = 30) #Y轴标签
plt.title("The comparison of training time among different target quantities",fontsize = 30) #标题

plt.show()