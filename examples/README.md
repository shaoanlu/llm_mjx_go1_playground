##Learning Notes
- Simulating LiDAR in mjx
  - although there is a `ray(...)` function implemented. It seems to be slow.
    - https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/_src/ray.py

```python
lidar_angles = np.linspace(0.0, 2 * np.pi, 360).reshape(-1, 1)
x_vec = np.cos(lidar_angles)
y_vec = np.sin(lidar_angles)
z_vec = np.zeros_like(x_vec)
vec = np.concatenate([x_vec, y_vec, z_vec], axis=1)
jvec = jax.numpy.array(vec)

state = env.reset(rng)

# robot forward direciton as x
dists, ids = ray_batch(m=env.env.mjx_model, d=state.data, pnt=state.data.qpos[:3]+jax.numpy.array([0.0, 0.0, 0.25]), vec=(state.data.xmat[1] @ jvec.T).T)

# world frame (not affested by robot orientation)
# dists, ids = ray_batch(m=env.env.mjx_model, d=state.data, pnt=state.data.qpos[:3]+jax.numpy.array([0.0, 0.0, 0.25]), vec=jvec)

plt.gca().set_aspect('equal')
for d, ang in zip(np.array(dists), lidar_angles):
  if d == -1 or abs(d) > 2:
    continue
  plt.plot([0, float(d*np.cos(ang))], [0, float(d*np.sin(ang))], "-", color="b")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()

```


- to try `RANGEFINDER`
  - rangefinder emits along +z axis
  - In the xml, we have to separately define new sites w/ proper rotations in body section and then attach rangefinder to it in the sensor section
  - lift the rangefinder a little bit (+0.1) in z-axis (to avoid self-detection?)
  - MJX does not support collision between box and cylinder atm (which is used in the navigation example)