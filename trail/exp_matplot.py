import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Your plotting code here
ax.plot([1, 2, 3], [4, 5, 6], [7, 8, 9])

plt.savefig("3d_plot.png")
