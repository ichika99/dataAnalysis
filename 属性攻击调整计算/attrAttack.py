import matplotlib.pyplot as plt
import numpy as np

# 设定 k 的值
k = 1

# 创建 a 和 b 的值域
a = np.arange(1, 4, 0.1)
b = np.arange(-1, 1, 0.1)
a, b = np.meshgrid(a, b)

# 计算方案一和方案二的伤害值
d_plan1 = np.where(b < 1, k * a * (1 - b), 0)
d_plan2 = np.where(a > b, k * (a - b), 0)
delta = d_plan1 - d_plan2

# 创建图表
fig = plt.figure()

# 添加方案一的子图
ax1 = fig.add_subplot(221, projection="3d")
ax1.plot_surface(a, b, d_plan1, cmap="viridis")
ax1.set_title("乘算伤害")
ax1.set_xlabel("属性伤害(x)")
ax1.set_ylabel("属性抗性(y)")
ax1.set_zlabel("伤害倍率(z)")

# 添加方案二的子图
ax2 = fig.add_subplot(222, projection="3d")
ax2.plot_surface(a, b, d_plan2, cmap="viridis")
ax2.set_title("加算伤害")
ax2.set_xlabel("属性伤害(x)")
ax2.set_ylabel("属性抗性(y)")
ax2.set_zlabel("伤害倍率(z)")

# 添加方案二的子图
ax3 = fig.add_subplot(212, projection="3d")
ax3.plot_surface(a, b, delta, cmap="plasma")
ax3.set_title("两种方案差异")
ax3.set_xlabel("属性伤害(x)")
ax3.set_ylabel("属性抗性(y)")
ax3.set_zlabel("伤害倍率差异(z)")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 显示图表
plt.show()
