import pandas as pd
import numpy as np

"""
把我们的结果从csv合并为excel

        csv --> excel

"""

csv_path = '/home/wyz/PycharmProjects/self-criticism-network-based-mult' \
           'i-target--visual-navigation/output_record/my_architecture_roa.csv'

data_pd = pd.read_csv(csv_path)

attribute_one = data_pd.columns[0]
attribute_two = data_pd.columns[1]

print(data_pd.columns)

print('attribute_one',attribute_one)
print('attribute_two',attribute_two)

new_columns = []

for item in data_pd[data_pd[attribute_one]==1.0].index:
    print("item",item)
    quantity_str = data_pd.loc[item-2,:]['targets_quantity']
    new_columns.append(quantity_str)

result_dict = dict()

for item in data_pd[data_pd[attribute_one]==1.0].index:
    quantity_str = data_pd.loc[item-2,:]['targets_quantity']
    print("QUANTITY:",quantity_str,'||  INDEX',item)
    # item ~ item+999
    result = data_pd.loc[item:item+998,:]['targets_quantity'].tolist()
    result_dict[quantity_str] = result


print(result_dict.keys())

for item in result_dict.keys():
    print(item,len(result_dict[item]))


result_pd = pd.DataFrame(result_dict,columns=result_dict.keys())

xlsx_path ='/home/wyz/PycharmProjects/self-criticism-network-based-multi-target--visual-navigation/output_record/my_architecture_roa.xlsx'


result_pd.to_excel(xlsx_path)