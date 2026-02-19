"""
Quick diagnostic: compare MuJoCo FK body positions with BVH FK positions
for the T-pose frame. If they match, the qpos->euler conversion is correct.
If not, we know exactly where things diverge.
"""
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.flomo_bvh import load_flomo_bvh

BVH_FILE = r"F:\experiments\tests202601\Flomo\data\timi\xs_20251101_aleblanc_lantern_nav-003.fbx.bvh"

frames, human_height, frame_time = load_flomo_bvh(BVH_FILE, skip_first_frame=False)
retargeter = GMR(src_human="bvh_flomo", tgt_robot="geno", actual_human_height=human_height, verbose=False)
model = retargeter.model
data = mj.MjData(model)

# retarget just frame 0 (T-pose)
qpos = retargeter.retarget(frames[0]).copy()
data.qpos[:] = qpos
mj.mj_forward(model, data)

num_joints = model.nbody - 1
joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
parents = [model.body_parentid[i] - 1 if model.body_parentid[i] > 0 else -1
           for i in range(1, model.nbody)]

# get MuJoCo world positions, convert to BVH Y-up cm
mj_positions_bvh = {}
for i in range(1, model.nbody):
    name = joint_names[i - 1]
    xp = data.xpos[i]
    mj_positions_bvh[name] = np.array([xp[0] * 100, xp[2] * 100, -xp[1] * 100])

# also get MuJoCo world orientations in Z-up
mj_world_quats = {}
for i in range(1, model.nbody):
    name = joint_names[i - 1]
    xq = data.xquat[i]  # wxyz
    mj_world_quats[name] = xq.copy()

# now convert qpos to BVH euler (variant B: no coord transform, intrinsic ZYX)
offsets = []
for i in range(1, model.nbody):
    mj_pos = model.body_pos[i]
    offsets.append(np.array([mj_pos[0] * 100, mj_pos[2] * 100, -mj_pos[1] * 100]))

bvh_rots_B = np.zeros((num_joints, 3))
root_pos_bvh = np.array([qpos[0] * 100, qpos[2] * 100, -qpos[1] * 100])

# root quat: just wxyz->xyzw for scipy, no coord transform
bvh_rots_B[0] = R.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler('ZYX', degrees=True)

q_idx = 7
for j in range(1, num_joints):
    w, x, y, z = qpos[q_idx], qpos[q_idx+1], qpos[q_idx+2], qpos[q_idx+3]
    bvh_rots_B[j] = R.from_quat([x, y, z, w]).as_euler('ZYX', degrees=True)
    q_idx += 4

# variant A: with coord transform (conjugation)
bvh_rots_A = np.zeros((num_joints, 3))
w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
bvh_rots_A[0] = R.from_quat([x, z, -y, w]).as_euler('ZYX', degrees=True)

q_idx = 7
for j in range(1, num_joints):
    w, x, y, z = qpos[q_idx], qpos[q_idx+1], qpos[q_idx+2], qpos[q_idx+3]
    bvh_rots_A[j] = R.from_quat([x, z, -y, w]).as_euler('ZYX', degrees=True)
    q_idx += 4

# BVH forward kinematics to get world positions
def bvh_fk(root_pos, euler_angles, offsets, parents):
    """Simple BVH FK: euler ZYX intrinsic -> world positions."""
    n = len(parents)
    world_rots = [None] * n
    world_poss = [None] * n

    # root
    world_rots[0] = R.from_euler('ZYX', euler_angles[0], degrees=True)
    world_poss[0] = root_pos.copy()

    for i in range(1, n):
        p = parents[i]
        local_rot = R.from_euler('ZYX', euler_angles[i], degrees=True)
        world_rots[i] = world_rots[p] * local_rot
        world_poss[i] = world_poss[p] + world_rots[p].apply(offsets[i])

    return world_poss

bvh_pos_B = bvh_fk(root_pos_bvh, bvh_rots_B, offsets, parents)
bvh_pos_A = bvh_fk(root_pos_bvh, bvh_rots_A, offsets, parents)

# world-FK approach: get world rots from MuJoCo, derive local BVH rots
# correct conjugation: R_bvh_world = M * R_mj_world * M^T
M_to_bvh = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
coord_rot = R.from_matrix(M_to_bvh)

bvh_rots_W = np.zeros((num_joints, 3))
world_rots_bvh = {}
for i in range(model.nbody):
    xq = data.xquat[i]  # wxyz
    r_mj = R.from_quat([xq[1], xq[2], xq[3], xq[0]])
    world_rots_bvh[i] = coord_rot * r_mj * coord_rot.inv()

# root
bvh_rots_W[0] = world_rots_bvh[1].as_euler('ZYX', degrees=True)
for j in range(1, num_joints):
    body_idx = j + 1
    parent_body_idx = model.body_parentid[body_idx]
    local_rot = world_rots_bvh[parent_body_idx].inv() * world_rots_bvh[body_idx]
    bvh_rots_W[j] = local_rot.as_euler('ZYX', degrees=True)

bvh_pos_W = bvh_fk(root_pos_bvh, bvh_rots_W, offsets, parents)

# also try: get MJ local rotations from FK, use those directly (no euler needed)
# this bypasses the qpos parsing entirely
bvh_rots_FK = np.zeros((num_joints, 3))
for j in range(num_joints):
    body_idx = j + 1
    parent_body_idx = model.body_parentid[body_idx]
    # get MJ world rotations
    pq = data.xquat[parent_body_idx]
    cq = data.xquat[body_idx]
    parent_r = R.from_quat([pq[1], pq[2], pq[3], pq[0]])
    child_r = R.from_quat([cq[1], cq[2], cq[3], cq[0]])
    # local rotation in MJ frame
    local_mj = parent_r.inv() * child_r
    # conjugate to BVH frame
    local_bvh = coord_rot * local_mj * coord_rot.inv()
    bvh_rots_FK[j] = local_bvh.as_euler('ZYX', degrees=True)

bvh_pos_FK = bvh_fk(root_pos_bvh, bvh_rots_FK, offsets, parents)

# compare
key_joints = ["Hips", "Spine", "Spine3", "LeftShoulder", "LeftArm", "LeftForeArm",
              "LeftHand", "RightShoulder", "RightArm", "RightHand",
              "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]

print("\n=== T-POSE FRAME: MuJoCo FK vs BVH FK world positions (BVH Y-up, cm) ===")
print(f"{'Joint':<20} {'MuJoCo FK':>35} {'BVH-B (no xform)':>35} {'BVH-A (xform)':>35} {'BVH-W (world FK)':>35}")
print("-" * 165)
for name in key_joints:
    idx = joint_names.index(name)
    mj_p = mj_positions_bvh[name]
    bp_B = bvh_pos_B[idx]
    bp_A = bvh_pos_A[idx]
    bp_W = bvh_pos_W[idx]
    print(f"{name:<20} ({mj_p[0]:8.2f},{mj_p[1]:8.2f},{mj_p[2]:8.2f}) "
          f"({bp_B[0]:8.2f},{bp_B[1]:8.2f},{bp_B[2]:8.2f}) "
          f"({bp_A[0]:8.2f},{bp_A[1]:8.2f},{bp_A[2]:8.2f}) "
          f"({bp_W[0]:8.2f},{bp_W[1]:8.2f},{bp_W[2]:8.2f})")

print("\n=== Position errors (cm) vs MuJoCo ground truth ===")
print(f"{'Joint':<20} {'err B':>10} {'err A':>10} {'err W':>10} {'err FK':>10}")
print("-" * 65)
for name in key_joints:
    idx = joint_names.index(name)
    mj_p = mj_positions_bvh[name]
    err_B = np.linalg.norm(bvh_pos_B[idx] - mj_p)
    err_A = np.linalg.norm(bvh_pos_A[idx] - mj_p)
    err_W = np.linalg.norm(bvh_pos_W[idx] - mj_p)
    err_FK = np.linalg.norm(bvh_pos_FK[idx] - mj_p)
    print(f"{name:<20} {err_B:10.3f} {err_A:10.3f} {err_W:10.3f} {err_FK:10.3f}")

# also print the actual euler angles for a couple key joints
print("\n=== Euler angles (deg) for key joints ===")
print(f"{'Joint':<20} {'B (no xform)':>30} {'A (xform)':>30} {'W (world FK)':>30} {'FK (conjugation)':>30}")
for name in ["Hips", "LeftShoulder", "LeftArm", "LeftHand", "RightShoulder", "RightArm"]:
    idx = joint_names.index(name)
    print(f"{name:<20} ({bvh_rots_B[idx][0]:8.2f},{bvh_rots_B[idx][1]:8.2f},{bvh_rots_B[idx][2]:8.2f}) "
          f"({bvh_rots_A[idx][0]:8.2f},{bvh_rots_A[idx][1]:8.2f},{bvh_rots_A[idx][2]:8.2f}) "
          f"({bvh_rots_W[idx][0]:8.2f},{bvh_rots_W[idx][1]:8.2f},{bvh_rots_W[idx][2]:8.2f}) "
          f"({bvh_rots_FK[idx][0]:8.2f},{bvh_rots_FK[idx][1]:8.2f},{bvh_rots_FK[idx][2]:8.2f})")

# also compare source input positions with MuJoCo IK output
print("\n=== Source input vs MuJoCo IK output (Z-up, meters) ===")
print(f"{'Joint':<20} {'Source pos':>30} {'MuJoCo FK pos':>30} {'Error(m)':>10}")
for name in key_joints:
    if name in frames[0]:
        src_pos = frames[0][name][0]
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            mj_pos = data.xpos[body_id]
            err = np.linalg.norm(src_pos - mj_pos)
            print(f"{name:<20} ({src_pos[0]:8.4f},{src_pos[1]:8.4f},{src_pos[2]:8.4f}) "
                  f"({mj_pos[0]:8.4f},{mj_pos[1]:8.4f},{mj_pos[2]:8.4f}) {err:10.4f}")
