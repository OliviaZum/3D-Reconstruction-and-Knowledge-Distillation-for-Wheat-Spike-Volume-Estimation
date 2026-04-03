import numpy as np
import matplotlib.pyplot as plt
import torch

cache = torch.load("local_stuff_experiments/dmap_real_default/dmap_cache_test.pth", weights_only=True)
print(cache.shape)

idx = 27
pc = cache[idx]

# remove zero padding
mask = ~(pc == 0).all(dim=1)
pc = pc[mask]

xyz = pc[:, 0:3].cpu()


# ---- Plot ----
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    s=0.5,          # much smaller
    c='red',
    depthshade=False
)

# remove everything
ax.set_axis_off()
ax.grid(False)

# equal scaling (important)
max_range = (xyz.max(dim=0).values - xyz.min(dim=0).values).max() / 2
mid = xyz.mean(dim=0)

print("Range:", xyz.max(dim=0).values - xyz.min(dim=0).values)
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

ax.view_init(elev=160, azim=160)

# remove margins
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig("spike_0_red.png", dpi=600)
plt.close()

############################################################
cache = torch.load("local_stuff_experiments/ply_cache_voxelized/ply_test.pth", weights_only=True)
print(cache.shape)

idx = 17
pc = cache[idx]

# remove zero padding
mask = ~(pc == 0).all(dim=1)
pc = pc[mask]

xyz = pc[:, 0:3].cpu()

# ---- Plot ----
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    s=5,          # much smaller
    c='red',
    depthshade=False
)

# remove everything
ax.set_axis_off()
ax.grid(False)

# equal scaling (important)
max_range = (xyz.max(dim=0).values - xyz.min(dim=0).values).max() / 2
mid = xyz.mean(dim=0)
print("Range:", xyz.max(dim=0).values - xyz.min(dim=0).values)

ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

ax.view_init(elev=210, azim=160)

# remove margins
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig("spike_0_red_full.png", dpi=600)
plt.close()