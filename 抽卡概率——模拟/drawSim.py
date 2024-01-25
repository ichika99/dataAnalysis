import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math


def simulate_gacha(
    card_types,
    probabilities,
    num_simulations,
    dynamic_prop_start,
    dynamic_prop_add,
    count_item,
):
    # 验证概率总和为1
    if not np.isclose(np.sum(probabilities), 1):
        raise ValueError("概率总和必须为1")

    results = {card: 0 for card in card_types}
    not_drawn_counts = [0] * len(card_types)  # 初始化每张卡片未抽中的次数
    actual_probabilities = probabilities.copy()
    calculate_probabilities = probabilities.copy()
    result_count = [0] * (
        math.ceil(
            (1 - probabilities[count_item - 1]) / dynamic_prop_add[count_item - 1]
            + dynamic_prop_start[count_item - 1]
        )
        + 1
    )

    # 进行模拟
    for _ in range(num_simulations):
        sum_probabilities = sum(actual_probabilities)

        # 权重调整
        if sum_probabilities > 1:
            delta = sum_probabilities - 1
            for i in range(len(actual_probabilities) - 1, -1, -1):
                if actual_probabilities[i] >= delta:
                    actual_probabilities[i] -= delta
                    break
                else:
                    delta -= actual_probabilities[i]
                    actual_probabilities[i] = 0

        elif sum_probabilities < 1:
            delta = 1 - sum_probabilities
            for i in range(len(actual_probabilities)):
                if calculate_probabilities[i] - actual_probabilities[i] >= delta:
                    actual_probabilities[i] += delta
                    break
                else:
                    delta -= calculate_probabilities[i] - actual_probabilities[i]
                    actual_probabilities[i] = calculate_probabilities[i]

        # 根据概率抽取卡片
        chosen_card = np.random.choice(card_types, p=actual_probabilities)
        chosen_index = card_types.index(chosen_card)
        # 记录结果
        results[chosen_card] += 1
        if chosen_index == count_item - 1:
            result_count[not_drawn_counts[chosen_index] + 1] += 1

        for i in range(len(card_types)):
            if i >= chosen_index:
                not_drawn_counts[i] = 0
                calculate_probabilities[i] = probabilities[i]
                actual_probabilities[i] = min(
                    calculate_probabilities[i], actual_probabilities[i]
                )
            else:
                not_drawn_counts[i] += 1
                if not_drawn_counts[i] >= dynamic_prop_start[i]:
                    calculate_probabilities[i] += dynamic_prop_add[i]
                    actual_probabilities[i] = calculate_probabilities[i]

    result_prob = [0] * len(result_count)
    result_distri = [0] * len(result_count)
    total_count = sum(result_count)
    current_distri = 0

    for i in range(len(result_count)):
        result_prob[i] = result_count[i] / total_count
        current_distri += result_prob[i]
        result_distri[i] = current_distri

    result_dis = pd.DataFrame(
        {
            "抽中次数": np.arange(0, len(result_count), 1),
            "样本数量": result_count,
            "概率密度": result_prob,
            "概率分布": result_distri,
        }
    ).set_index("抽中次数")
    # 结果处理
    result_df = pd.DataFrame(list(results.items()), columns=["道具", "抽中次数"])
    return result_df, result_dis


# 卡片种类、概率和模拟次数的例子
card_types = ["SSR", "SR", "R", "N"]
probabilities = [0.003, 0.01, 0.087, 0.9]
dynamic_prop_start = [60, 14, 8, 0]
dynamic_prop_add = [0.049875, 0.1645, 0.4565, 0]
count_item = 1
num_simulations = 100000
temp, aaa = simulate_gacha(
    card_types,
    probabilities,
    num_simulations,
    dynamic_prop_start,
    dynamic_prop_add,
    count_item,
)

pd.set_option("display.max_rows", None)

# 运行模拟
result1, result2 = simulate_gacha(
    card_types,
    probabilities,
    num_simulations,
    dynamic_prop_start,
    dynamic_prop_add,
    count_item,
)

newDf = pd.merge(result1, temp, on="道具", suffixes=("_r1", "_r3"), how="outer")
newDf.fillna(0, inplace=True)
newDf["抽中次数"] = newDf["抽中次数_r1"] + newDf["抽中次数_r3"]
newDf = newDf[["道具", "抽中次数"]]
print(newDf)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 使用seaborn进行可视化
# sns.set_theme(style="whitegrid")
# sns.barplot(x=result2.index, y="概率密度", data=result2)

# plt.bar(result2.index, result2["概率密度"], width=1, alpha=0.6, color='b')
# plt.step(result2.index, result2["概率分布"], where='post', color='r', label='CDF')

plt.plot(result2.index, result2["概率密度"], aa=True)
plt.title(card_types[count_item] + "的所需抽数概率密度")
plt.xlabel("抽数")

plt.plot(result2.index, result2["概率分布"], aa=True)
plt.title(card_types[count_item] + "的所需抽数概率分布")
plt.xlabel("抽数")

# 饼图
# plt.pie(result1["抽中次数"],labels=result1["道具"],colors=sns.color_palette("pastel"),autopct='%.2f%%')
# plt.title('概率图')
plt.show()
