# -*- coding:utf-8 -*-
# 开发者：北海
# 设备：Dell
# 创建时间：2025/2/18  10:21
# 开发环境:PyCharm


from mp_api.client import MPRester
import csv
from tqdm import tqdm

API_KEY = "igRHy7zYOKzWD18jY76XtjbwEpRl6SoH"
mpr = MPRester(API_KEY)

docs = mpr.summary.search(elements=['Si'],fields=['material_id',
                                                  'structure',
                                                  'formation_energy_per_atom',
                                                  'band_gap'])
'''                                                   
for idoc in docs[:5000]:  # 保存前5000个结构到cif文件
    print(idoc.material_id)
    idoc.structure.to(str(idoc.material_id)+".cif")
'''

# 打开一个新的文件用于写入，如果文件不存在则创建
with open('Si_data/id_prop.csv', 'w', newline='') as csvfile:
    # 创建一个csv写入器
    writer = csv.writer(csvfile)
    # 写入标题行（可选）
    # writer.writerow(['ID', 'Property'])
    # 遍历列表，写入每一行数据
    for idoc in tqdm(docs[:5000]):  # 保存前5000个结构到cif文件
        # 写入ID和band_gap数据方便后续训练
        writer.writerow([idoc.material_id, idoc.formation_energy_per_atom])