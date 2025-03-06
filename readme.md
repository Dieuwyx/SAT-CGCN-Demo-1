# 思路1：CGCN嵌套SAT
### 2025年2月20日
### 方法
修改 CGCNN 的读取和训练文件，使之能够运行 SAT 的模型，其中需要做出以下修改：  
1. data 文件保持不变，传入模型中需要做出改变。所以需要修改SAT文件夹中的data文件，
让它与CGCNN-data相对应。




### 日志

#### 2025年2月20日log：
1.  修改了sat.data文件，使之能够读取CIF文件。
2. 

#### 2025年2月25日log：
1. 方案1失败

# 思路2：SAT嵌套CGCN
#### 2025年2月25日
### 方法
1. 保持 CGCN 框架不变，使用 SAT 的模型
2. 修改  CGCN 的 data 文件，使其能具备 SAT 的子图分析能力
3. 修改 main 文件，使其能够使用 SAT 的神经网络进行预测

### 日志
#### 2025年2月27日
1. 修改了dataset的获取方式，见Get_Si_data.ipynb
2. 在dataset中添加了gitignore
3. 重新设计了CGCN参数，训练Si数据得出了较好的结果
#### 2025年3月5日
1. 修改了main_demo_0.py可以使用固定的参数了，并进行简单的训练了
2. 新建了CSAT文件夹，包含修改后的data、model文件
3. 修改了data文件，在collect_pool中添加了子图检索的矫正，在CIF函数中添加了Khop子图的提取功能
4. 修改了model文件，新增了satlayer，更新了晶体net的迭代结构
#### 2025年3月6日
1. 新建了SAT_demo初步对SAT进行了测试
2. 新建了csat_demo_1项目，其结构如下：
```text
# 文件结构：
sat4crystals/
├── data_loader.py       # 晶体数据加载与图转换
├── sat_model.py         # SAT模型定义
├── train.py             # 训练脚本
└── requirements.txt     # 依赖库
```
3. 构建了CGCN的简易项目：CGCN_Main_easy.py以及cgcnn文件已更新
4. 对参数进行了打印如下：
```text
orig_atom_fea_len 92
nbr_fea_len 41
atom_fea_len 64
```

# 2025年3月8日要做的事情
1. 在cgcnn.data中添加函数crystal_to_graph函数
2. 新建一个train
3. 重构一下代码结构


`python main_demo_0.py ./dataset`