https://github.com/ichika99/dataAnalysis

# 抽卡模拟
用于计算目前手游常见抽卡模式的期望情况（伪随机，保底）
drawCardBySimulate：使用excel VBA进行模拟，通过大量模拟数据得出一个近似的抽卡分布曲线，可信度较高。
drawCardByMatrix：参考马尔可夫链，使用python求解转移矩阵，得到精确的抽卡概率。性能消耗比较大，实际手游抽卡品质多，保底次数高，想要求解需要较大的电脑内存
