import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# avg-user-cpu
x  = [i*100 for i in range(1, 10)]

y0 = [7.071, 21.473, 43.250, 72.444, 112.421, 158.388, 211.322, 269.988, 343.326]
y1 = [0.110, 0.229, 0.364, 0.504, 0.578, 0.699, 0.844, 0.949, 1.069]

plt.plot(x, y0, 'yo-', label='Bonawitz\'s secure aggregation')
plt.plot(x, y1, 'gs-', label='DemoFL')

plt.xlabel('Number of clients')
plt.ylabel('Average computation cost (ms)')

plt.legend()
plt.show()

# print(plt.style.available)