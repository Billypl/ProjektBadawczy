import numpy as np
import matplotlib.pyplot as plt

points = np.load(
    r"results\obstacle_pointcloud\step_5_6_test\000153\obstacle_points.npy"
)

# Ogranicz liczbę punktów, by wykres szybciej się renderował.
if len(points) > 50_000:
    rng = np.random.default_rng(42)
    points = points[rng.choice(len(points), 50_000, replace=False)]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    points[:, 0], points[:, 1], points[:, 2],
    c=points[:, 2], cmap="turbo", s=1
)

ax.set_xlabel("X [m]", labelpad=12)
ax.set_ylabel("Y [m]", labelpad=10)

# Look forward from the camera origin along +Z.
ax.view_init(elev=-90, azim=-90)
ax.set_proj_type("persp", focal_length=4.0)
ax.set_ylim(points[:, 1].min(), points[:, 1].max())
ax.set_zlim(0, points[:, 2].max())
ax.set_zticks([])
ax.set_zlabel("")
ax.set_box_aspect((
    np.ptp(points[:, 0]),
    np.ptp(points[:, 1]),
    points[:, 2].max(),
))

fig.colorbar(scatter, ax=ax, label="Głębokość [m]")
plt.show()