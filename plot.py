import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

x  = [0.01, 0.03, 0.05, 0.1]

y0  = [0.108, 0.383, 0.685, 1.367]
y1  = [0.271, 0.840, 1.419, 2.773]
y2  = [0.050, 0.170, 0.279, 0.539]

plt.plot(x, y2, 'r^-', label='Number of clients: 100')
plt.plot(x, y0, 'yo-', label='Number of clients: 250')
plt.plot(x, y1, 'gs-', label='Number of clients: 500')

plt.xlabel('Proportion of leaders')
plt.ylabel('Average computation cost (ms)')

plt.legend()
plt.show()

# print(plt.style.available)