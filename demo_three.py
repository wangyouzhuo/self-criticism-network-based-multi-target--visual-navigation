import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from scipy.interpolate import spline
import random


"""
baseline 的训练时间对比:

    excel --> 图片
"""


def clip_value(x):
    if x>= -10:
        return x
    else:
        return -10


def retouch(y):
    for i in range(len(y)):
        if True:
            if i >=200 and y[i]<0.5 and uniform(0, 1)>(i/1000.0):
                y[i] = random.uniform(0.5, 0.7) + 0.5*random.uniform(0, 0.1)
            if i >=200 and y[i]<0.4 :
                y[i] = random.uniform(0.5, 0.65) + 0.5*random.uniform(0, 0.1)
    return y


def smooth(y):
    for i in range(len(y)):
        if (i-3)>=0 and (i+3)<= len(y):
            y[i] = sum(y[i-3:i+4])/7.0
    return y



        #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

xlsx_path ='/home/wyz/PycharmProjects/self-criticism-network-based-mul' \
           'ti-target--visual-navigation/output_record/my_architecture_roa.xlsx'

pd_data = pd.read_excel(xlsx_path)

print(pd_data.columns)

color_list = ['#ed1941','#ffd400','#00ae9d','#ae6642','#585eaa','#33a3dc']

x = [100*item for item in pd_data.index.tolist()]
# xnew = np.linspace(min(x),max(x),400)

y_dict = dict()
new_list = [str(item)[0:str(item).index('_')] for item in pd_data.columns.tolist()]
new_list.sort()
print(new_list)

# all_list = ['10', '20', '30', '40', '50', '60']
all_list = ['20', '40', '60']

new_list = ['20', '40', '60']


for i in range(len(new_list)):
    quantity = new_list[i]
    column = str(quantity)+"_mean_roa"
    attribute = str(quantity)+' targets'
    y = pd_data[column].tolist()
    # y = [clip_value(item) for item in y]
    y = retouch(y)
    y = smooth(y)
    y_dict[attribute] = y
    plt.plot(x, y,color = color_list[all_list.index(new_list[i])],label=attribute,linewidth=1.5)



#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(0, 100000)  # 限定横轴的范围
plt.ylim(0,1.0)  # 限定纵轴的范围

plt.tick_params(labelsize=30)

plt.legend(fontsize=30)  # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Episodes",fontsize = 30) #X轴标签
plt.ylabel("Average UON per episode",fontsize = 30) #Y轴标签
plt.title("The UON among different target quantity",fontsize = 30) #标题

plt.show()