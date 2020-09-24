import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

x1 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
y1 = [92.875, 95.458, 96.841, 97.545, 97.868, 98.083, 98.31, 98.37, 98.48, 98.53, 98.54]

x2 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
y2 = [92.690, 95.318, 96.764, 97.453, 97.821, 98.078, 98.24, 98.33,  98.47, 98.44, 98.52]

x3 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
y3 = [63.658, 83.891, 88.333, 91.720, 87.246, 89.631, 96.00, 95.40, 95.50, 94.97, 94.66]

x4 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
y4 = [61.026, 68.355, 79.713, 85.812, 92.818, 92.056, 95.86, 95.97, 96.18, 94.79, 94.16]


plt.plot(x1, y1, 'b.-', label='DemoFL with IID data')
plt.plot(x2, y2, 'r,-', label='FedAvg with IID data')
plt.plot(x3, y3, 'ys-', label='DemoFL with non-IID data')
plt.plot(x4, y4, 'g^-', label='FedAvg with non-IID data')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

# print(plt.style.available)