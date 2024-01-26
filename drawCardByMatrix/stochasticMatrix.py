import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import solve


def calculate_transition_probabilities():
    # 定义状态空间的大小
    max_state = 11  # 我们考虑连续未中奖的次数最多为10次
    matrix_size = max_state * max_state

    # 初始化转移矩阵
    transition_matrix = np.zeros((matrix_size, matrix_size))

    # 定义初始概率
    initial_prob_first_prize = 0.05
    initial_prob_second_prize = 0.15

    # 遍历每个状态
    for x in range(max_state):
        for y in range(max_state):
            # 计算当前状态的一等奖和二等奖概率
            prob_first_prize = min(initial_prob_first_prize + 0.15 * max(0, x - 9), 1)
            prob_second_prize = min(initial_prob_second_prize + 0.20 * max(0, y - 5), 1)

            # 确保概率总和不超过100%
            total_prob = prob_first_prize + prob_second_prize
            if total_prob > 1:
                prob_second_prize = 1 - prob_first_prize

            # 计算转移概率
            current_state_index = x * max_state + y
            prob_no_prize = 1 - prob_first_prize - prob_second_prize

            # 转移到下一个状态
            next_state_index_no_prize = min(x + 1, 10) * max_state + min(y + 1, 10)
            next_state_index_first_prize = y * max_state  # 一等奖中奖，二等奖未中奖次数不变
            next_state_index_second_prize = (y + 1) % max_state  # 二等奖中奖，一等奖未中奖次数重置为0

            # 更新转移矩阵
            transition_matrix[
                current_state_index, next_state_index_no_prize
            ] = prob_no_prize
            transition_matrix[
                current_state_index, next_state_index_first_prize
            ] = prob_first_prize
            transition_matrix[
                current_state_index, next_state_index_second_prize
            ] = prob_second_prize

    return transition_matrix


# 计算转移矩阵
transition_matrix = calculate_transition_probabilities()

df = pd.DataFrame(transition_matrix)

# 设置行和列的标签
max_state = 11
state_labels = [f"({i},{j})" for i in range(max_state) for j in range(max_state)]
df.index = state_labels
df.columns = state_labels

df


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


pi = calculate_steady_state(transition_matrix)

print(pi)
