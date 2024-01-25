import numpy as np
import pandas as pd
from scipy.linalg import solve
import itertools
import datetime
import os

cards = (
    {
        "id": 1,
        "name": "红卡",
        "weight": 10,
        "start_add_prob": 6,
        "value_add_prob": 0.2,
    },
    {
        "id": 2,
        "name": "红卡",
        "weight": 30,
        "start_add_prob": 4,
        "value_add_prob": 0.3,
    },
    {"id": 3, "name": "高橙", "weight": 150, "start_add_prob": 0, "value_add_prob": 0},
)

matrix_size = 1

# 计算转移矩阵
def calculate_stochastic_matrix(cardList):

    print("数据预处理中...")

    global matrix_size  # 在函数中使用 global 关键字声明 matrix_size
    card_data = pd.DataFrame(cardList).sort_values(by="id")
    card_data["base_prob"] = 0
    card_data["max_time"] = 0
    card_data["temp_time"] = 0
    card_data["temp_prob"] = 0

    # 计算总权重
    sum_weight = card_data['weight'].sum()

    def get_base_data(row, sum_weight):
        global matrix_size
        base_prob = row["weight"] / sum_weight
        # 未启用动态概率处理
        if row["value_add_prob"] <= 0:
            max_time = 1
        else:
            max_time = (1 - base_prob) / row["value_add_prob"] + row["start_add_prob"]
            max_time = int(max_time) if max_time == int(max_time) else int(max_time) + 1
        matrix_size *= max_time
        return base_prob, max_time

    card_data[['base_prob', 'max_time']] = card_data.apply(lambda row: get_base_data(row, sum_weight), axis=1, result_type='expand')

    card_data["temp_prob"] = card_data["base_prob"]

    transition_matrix = np.zeros((matrix_size, matrix_size))

    # 生成转移矩阵
    print(f"计算转移矩阵，矩阵大小：{matrix_size}*{matrix_size}...")
    calculate_matrix_cell(card_data, transition_matrix, 0, 0)
    
    result1=pd.DataFrame(transition_matrix)
    state_labels = list(itertools.product(*[range(0, int(x)) for x in card_data['max_time']]))

    result1.index = state_labels
    result1.columns = state_labels

    # 稳态分布计算
    print(f"计算稳态分布，状态数：{matrix_size}...")
    steady_state=calculate_steady_state(transition_matrix)

    # 数学期望计算

    result2=pd.Series(steady_state)
    result2.index=state_labels
    
    return result1,result2


# 抽卡表，转移矩阵，当前奖品序号，当前总概率
def calculate_matrix_cell(data: pd.DataFrame, matrix, num, current_prob):
    current_item_prob = 0  # 当前类型奖品在总权重中占比

    for i in range(0, int(data.iloc[num].max_time)):
        # 当前总概率不足1，则需要用当前奖品概率补足
        if current_prob - current_item_prob < 1:
            if i < data.iloc[num].start_add_prob:
                calculate_prob = data.iloc[num].base_prob
            else:
                calculate_prob = (
                    data.iloc[num].base_prob
                    + (i - data.iloc[num].start_add_prob + 1)
                    * data.iloc[num].value_add_prob
                )

            # 计算当前奖品的实际概率与总概率
            current_prob -= current_item_prob
            current_item_prob = min(1 - current_prob, calculate_prob)
            current_prob += current_item_prob

        # 写入当前概率与未抽中次数
        data.loc[data.index[num], 'temp_time'] = i
        data.loc[data.index[num], 'temp_prob'] = current_item_prob

        # 最末奖品，处理转移矩阵
        if num == len(data)-1:
            # 计算当前所处状态与下一状态序号
            current_state_index=0
            for row in data.itertuples():
                current_state_index=current_state_index*row.max_time+row.temp_time
            
            # 计算下一状态的状态序号并输出值
            base_index=len(matrix) #基础系数，用于分配每个中奖情况的系数区域分布
            next_state_index=0
            for row in data.itertuples():
                matrix[int(current_state_index),int(next_state_index)]=row.temp_prob
                
                base_index/=row.max_time
                next_state_index+=(row.temp_time+1)*base_index

                if(next_state_index)==len(matrix): return
                 
            return

        # 非最末奖品，向下一层奖品迭代
        else:
            calculate_matrix_cell(data, matrix, num + 1, current_prob)

# 稳态分布计算
def calculate_steady_state(P):
    # 构造线性方程组
    dim = P.shape[0]
    A = np.transpose(P - np.eye(dim))
    A[-1] = np.ones(dim)  # 替换最后一行为和为1的条件
    b = np.zeros(dim)
    b[-1] = 1

    # 求解线性方程组
    pi = solve(A, b)
    return pi

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data=pd.read_excel('data.xlsx','data')

transition_matrix,steady_state=calculate_stochastic_matrix(data)

print("写入结果...")
if len(transition_matrix)>2000:print("矩阵太大了，仅保留部分结果！")

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
with pd.ExcelWriter(current_time+'.xlsx') as writer:
    transition_matrix.iloc[:2000,:2000].to_excel(writer,sheet_name="转移矩阵")
    steady_state.to_excel(writer,sheet_name="稳态分布")

os.startfile(current_time+'.xlsx')
