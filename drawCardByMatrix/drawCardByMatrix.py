import numpy as np
import pandas as pd
from scipy.linalg import solve
import itertools
import datetime
import os

matrix_size = 1


def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    data = pd.read_excel("data.xlsx", "data")

    transition_matrix, steady_state, card_data = calculate_stochastic_matrix(data)

    # 结果输出为excel，考虑矩阵大小适量优化
    print("写入结果...")
    if len(transition_matrix) > 500:
        print("矩阵太大了，仅保留部分结果！")

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with pd.ExcelWriter("./result/" + current_time + ".xlsx") as writer:
        transition_matrix.iloc[:500, :500].to_excel(writer, sheet_name="转移矩阵")
        steady_state.to_excel(writer, sheet_name="稳态分布")
        card_data[
            [
                "id",
                "name",
                "weight",
                "start_add_prob",
                "value_add_prob",
                "base_prob",
                "max_time",
                "mathematical_expectation",
            ]
        ].to_excel(writer, sheet_name="数据结果", index=False)

    os.startfile(os.path.abspath("./result/" + current_time + ".xlsx"))


# 计算流程
def calculate_stochastic_matrix(cardList):
    global matrix_size

    # 预处理基础数据
    print("数据预处理中...")
    card_data = calculate_base_data(cardList)

    # 生成转移矩阵
    print(f"计算转移矩阵，矩阵大小：{matrix_size}*{matrix_size}...")
    transition_matrix = np.zeros((matrix_size, matrix_size))
    calculate_matrix_cell(card_data, transition_matrix, 0, 0)

    # 稳态分布计算
    print(f"计算稳态分布，状态数：{matrix_size}...")
    steady_state = calculate_steady_state(transition_matrix)

    # 数学期望计算
    print("计算数学期望...")
    calculate_mathematical_expectation(card_data, steady_state, 0, 0, matrix_size)

    # 整理返回数据
    result1 = pd.DataFrame(transition_matrix)

    state_labels = list(
        itertools.product(*[range(0, int(x)) for x in card_data["max_time"]])
    )

    result1.index = state_labels
    result1.columns = state_labels

    result2 = pd.Series(steady_state)
    result2.index = state_labels

    return result1, result2, card_data


# 基础数据的处理，保底次数等
def calculate_base_data(data):
    card_data = pd.DataFrame(data).sort_values(by="id")
    card_data["base_prob"] = 0  # 基础概率
    card_data["max_time"] = 0  # 最大保底次数
    card_data["temp_time"] = 0  # 迭代计算用的临时未中奖次数
    card_data["temp_prob"] = 0  # 迭代计算用的临时概率
    card_data["mathematical_expectation"] = 0.0  # 数学期望

    # 计算总权重
    sum_weight = card_data["weight"].sum()

    def get_base_data(row, sum_weight):
        global matrix_size
        base_prob = row["weight"] / sum_weight
        # 未启用动态概率的奖项处理
        if row["value_add_prob"] <= 0:
            max_time = 1
        else:
            max_time = (1 - base_prob) / row["value_add_prob"] + row["start_add_prob"]
            max_time = int(max_time) if max_time == int(max_time) else int(max_time) + 1
        matrix_size *= max_time
        return base_prob, max_time

    card_data[["base_prob", "max_time"]] = card_data.apply(
        lambda row: get_base_data(row, sum_weight), axis=1, result_type="expand"
    )
    card_data["temp_prob"] = card_data["base_prob"]

    return card_data


# 转移矩阵计算，具体每格内容与位置
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
        data.loc[data.index[num], "temp_time"] = i
        data.loc[data.index[num], "temp_prob"] = current_item_prob

        # 最末奖品，处理转移矩阵
        if num == len(data) - 1:
            # 计算当前所处状态与下一状态序号
            current_state_index = 0
            for row in data.itertuples():
                current_state_index = current_state_index * row.max_time + row.temp_time

            # 计算下一状态的状态序号并输出值
            base_index = len(matrix)  # 基础系数，用于分配每个中奖情况的系数区域分布
            next_state_index = 0
            for row in data.itertuples():
                matrix[int(current_state_index), int(next_state_index)] = row.temp_prob

                base_index /= row.max_time
                next_state_index += (row.temp_time + 1) * base_index

                if (next_state_index) == len(matrix):
                    return

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


# 计算数学期望
def calculate_mathematical_expectation(
    data, steady_state, num, current_state_index, max_index
):
    for i in range(0, int(data.iloc[num].max_time)):
        # 当前状态下该奖项未中奖次数为0，为中奖状态，写入概率
        if i == 0:
            data.loc[data.index[num], "mathematical_expectation"] += steady_state[
                int(current_state_index)
            ]

        # 当前状态下该奖项未中奖次数不为0，为未中奖状态，查询后续奖品中奖情况
        else:
            current_max_index = max_index / data.iloc[num].max_time
            next_state_index = current_state_index + i * current_max_index
            calculate_mathematical_expectation(
                data, steady_state, num + 1, next_state_index, current_max_index
            )


main()
