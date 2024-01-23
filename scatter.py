import matplotlib.pyplot as plt

# 示例数据
x = [0.00001,0.0001, 0.001, 0.01]
y = [40,55.78, 44.89, 50.48]

# 绘制散点图
plt.scatter(x, y, color='blue', marker='o', label='Scatter Plot')

# 添加标题和标签
plt.title('Simple Scatter Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# 添加图例
plt.legend()

# 显示图表
plt.show()