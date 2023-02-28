from plyfile import PlyData, PlyElement
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "./data/raw_points.ply"
    data = PlyData.read(filepath)

    x = data["vertex"]["x"]
    y = data["vertex"]["y"]
    z = data["vertex"]["z"]

    y -= np.min(y)
    pc = np.stack([x, z, y]).T
    print(pc.shape)
    f = plt.figure()
    ax = f.add_subplot(111, projection="3d")
    ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], "o", markersize=1)
    plt.show()
