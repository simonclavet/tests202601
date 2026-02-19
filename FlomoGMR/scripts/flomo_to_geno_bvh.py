"""
Retarget Flomo Xsens BVH to Geno/LAFAN1 skeleton BVH.

The pipeline has three stages:
  1. IK solve  — GMR's mink IK solver matches root, spine, and foot positions.
  2. Preset    — leg rotations are copied from source BEFORE IK, giving the
                 solver a good starting point for knee direction.
  3. Override  — arm, head, and foot rotations are copied from source AFTER IK,
                 bypassing whatever the solver produced for those joints.

Stages 2 and 3 both need "twist offsets" to compensate for the fact that Geno's
rest-pose bone directions differ from the source skeleton's T-pose directions.
For example, Geno's arm bones point straight up (+Z) in rest pose, while the
source T-pose has them horizontal. See compute_twist_offsets() for the math.
"""
import argparse
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.flomo_bvh import load_flomo_bvh


def bvh_save(filename, data):
    """Write a BVH file. Root gets 6 channels (pos + rot), others get 3 (rot only)."""
    channelmap_inv = {'x': 'Xrotation', 'y': 'Yrotation', 'z': 'Zrotation'}
    rots = data['rotations']
    poss = data['positions']
    offsets = data['offsets']
    parents = data['parents']
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0 / 30.0)

    with open(filename, 'w') as f:
        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, offsets[0][0], offsets[0][1], offsets[0][2]))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        # jseq tracks joint visit order during hierarchy write.
        # BVH motion data must follow this exact same order.
        jseq = [0]

        def save_joint(f, offsets, order, parents, names, t, i, jseq):
            jseq.append(i)
            f.write("%sJOINT %s\n" % (t, names[i]))
            f.write("%s{\n" % t)
            t += '\t'
            f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, offsets[i][0], offsets[i][1], offsets[i][2]))
            f.write("%sCHANNELS 3 %s %s %s \n" % (t,
                channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
            end_site = True
            for j in range(len(parents)):
                if parents[j] == i:
                    t, jseq = save_joint(f, offsets, order, parents, names, t, j, jseq)
                    end_site = False
            if end_site:
                f.write("%sEnd Site\n" % t)
                f.write("%s{\n" % t)
                t += '\t'
                f.write("%sOFFSET %.9g %.9g %.9g\n" % (t, 0.0, 0.0, 0.0))
                t = t[:-1]
                f.write("%s}\n" % t)
            t = t[:-1]
            f.write("%s}\n" % t)
            return t, jseq

        for i in range(len(parents)):
            if parents[i] == 0:
                t, jseq = save_joint(f, offsets, order, parents, names, t, i, jseq)

        t = t[:-1]
        f.write("%s}\n" % t)
        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots))
        f.write("Frame Time: %f\n" % frametime)

        for i in range(rots.shape[0]):
            for j in jseq:
                if j == 0:
                    f.write("%.9g %.9g %.9g %.9g %.9g %.9g " % (
                        poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                        rots[i, j, 0], rots[i, j, 1], rots[i, j, 2]))
                else:
                    f.write("%.9g %.9g %.9g " % (rots[i, j, 0], rots[i, j, 1], rots[i, j, 2]))
            f.write("\n")


def source_local_quat(model, frame_data, name, copy_parent):
    """
    Compute local joint quaternion from source global orientations.

    geno.xml has no body_quat attributes (all identity rest frames), so:
        joint_quat = inv(parent_world_rot) * child_world_rot

    Returns wxyz quaternion as numpy array, or None if joint not in source data.
    """
    parent_name = copy_parent[name]
    if name not in frame_data or parent_name not in frame_data:
        return None
    p_wxyz = frame_data[parent_name][1]
    c_wxyz = frame_data[name][1]
    p_rot = R.from_quat([p_wxyz[1], p_wxyz[2], p_wxyz[3], p_wxyz[0]])
    c_rot = R.from_quat([c_wxyz[1], c_wxyz[2], c_wxyz[3], c_wxyz[0]])
    local_xyzw = (p_rot.inv() * c_rot).as_quat()
    return np.array([local_xyzw[3], local_xyzw[0], local_xyzw[1], local_xyzw[2]])


def compute_twist_offsets(model, tpose_frame, override_joints):
    """
    Compute per-joint rotation offsets that compensate for differences in bone
    directions between the Geno rest pose and the source T-pose.

    THE PROBLEM:
    When we copy a local rotation from the source skeleton to Geno, the body gets
    the correct world ORIENTATION, but the bone DIRECTION (from joint to child)
    can be wrong because the child offset vector differs between skeletons.
    For example, Geno's LeftArm→LeftForeArm offset points along +Z (up), while
    the source T-pose has it pointing laterally. Copying an identity rotation from
    the source T-pose leaves Geno's arm pointing up — a ~90° error.

    THE FIX:
    For each joint, compute a "twist" rotation R_twist that maps the Geno bone
    direction to the source T-pose bone direction, both expressed in the joint's
    local frame.

    PER-FRAME APPLICATION:
    For a joint J with source local rotation q_src:
        q_corrected = inv(R_twist_parent) * q_src * R_twist_self

    This chains correctly through the hierarchy. After correction:
        world_rot_J_geno = world_rot_J_src * R_twist_J
    which means the bone direction in world space is:
        world_rot_J_geno * d_geno = world_rot_J_src * R_twist_J * d_geno
                                  = world_rot_J_src * d_target
    ...exactly matching the source skeleton's bone direction.

    The inv(R_twist_parent) factor undoes the parent's twist so it doesn't
    accumulate and distort children. Without it, each joint in a chain would
    get the parent's twist baked in on top of its own.

    Returns:
        twist_offsets: dict {joint_name: R} — the per-joint R_twist
        twist_parent:  dict {joint_name: R} — the parent's R_twist for chaining
    """
    # Geno rest-pose world positions (all joint_quats = identity)
    rest_data = mj.MjData(model)
    mj.mj_forward(model, rest_data)

    # each chain is processed parent-to-child so we can propagate R_twist
    override_chains = [
        ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'],
        ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'],
        ['Neck', 'Head'],
        ['LeftFoot', 'LeftToeBase'],
        ['RightFoot', 'RightToeBase'],
    ]

    # for hands, the source has no finger joints so we can't compute a hand→finger
    # bone direction. instead we use MiddleFinger1 as Geno's reference direction
    # and forearm→hand as the source's proxy direction (in T-pose the palm extends
    # in the same direction as the forearm).
    # format: {joint: (geno_child, source_from, source_to)}
    hand_ref = {
        'LeftHand': ('LeftHandMiddle1', 'LeftForeArm', 'LeftHand'),
        'RightHand': ('RightHandMiddle1', 'RightForeArm', 'RightHand'),
    }

    twist_offsets = {}
    twist_parent = {}
    for chain in override_chains:
        for ci in range(len(chain)):
            name = chain[ci]
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)

            # inherit parent's twist from previous joint in chain (identity if first)
            if ci > 0 and chain[ci - 1] in twist_offsets:
                twist_parent[name] = twist_offsets[chain[ci - 1]]
            else:
                twist_parent[name] = R.identity()

            # figure out what bone direction to align
            if name in hand_ref:
                geno_child, src_from, src_to = hand_ref[name]
                ref_child_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, geno_child)
                src_pair = (src_from, src_to)
            elif ci + 1 < len(chain):
                ref_child = chain[ci + 1]
                ref_child_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ref_child)
                src_pair = (name, ref_child)
            else:
                # true leaf (Head, LeftToeBase, etc.) — no child bone to align
                twist_offsets[name] = R.identity()
                continue

            # d_geno: bone direction in Geno rest pose.
            # since all orientations are identity, world positions == local positions,
            # so we can just subtract world coords directly.
            d_geno = rest_data.xpos[ref_child_id] - rest_data.xpos[body_id]
            if np.linalg.norm(d_geno) < 1e-6:
                twist_offsets[name] = R.identity()
                continue
            d_geno = d_geno / np.linalg.norm(d_geno)

            # d_target: source T-pose bone direction, transformed into J's local frame.
            # we use J's SOURCE world orientation to go from world to local because
            # R_twist bridges from the source frame to the geno frame.
            src_from_name, src_to_name = src_pair
            if src_from_name not in tpose_frame or src_to_name not in tpose_frame:
                twist_offsets[name] = R.identity()
                continue
            d_src_world = tpose_frame[src_to_name][0] - tpose_frame[src_from_name][0]
            if np.linalg.norm(d_src_world) < 1e-6:
                twist_offsets[name] = R.identity()
                continue
            d_src_world = d_src_world / np.linalg.norm(d_src_world)
            j_wxyz = tpose_frame[name][1]
            j_rot = R.from_quat([j_wxyz[1], j_wxyz[2], j_wxyz[3], j_wxyz[0]])
            d_target = j_rot.inv().apply(d_src_world)

            # minimum-angle rotation from d_geno to d_target
            twist_offsets[name], _ = R.align_vectors([d_target], [d_geno])

    # make sure every override joint has an entry (even if not in any chain)
    for name in override_joints:
        if name not in twist_offsets:
            twist_offsets[name] = R.identity()
        if name not in twist_parent:
            twist_parent[name] = R.identity()

    return twist_offsets, twist_parent


def qpos_to_bvh(model, qpos_array, frame_time):
    """
    Convert MuJoCo qpos array to BVH-compatible data.

    Coordinate transform: generate_geno_xml.py mapped BVH(x,y,z) → MJ(x,-z,y).
    To reverse: MJ(x,y,z) → BVH(x, z, -y), then ×100 for cm.

    For quaternions (w,x,y,z), the imaginary part IS the rotation axis — a vector
    that must be remapped the same way as positions:
        MJ quat (w, vx, vy, vz) → BVH quat (w, vx, vz, -vy)

    Euler convention: BVH "Zrotation Yrotation Xrotation" means intrinsic ZYX.
    Must use uppercase 'ZYX' (intrinsic) in scipy, NOT lowercase 'zyx' (extrinsic).
    """
    num_joints = model.nbody - 1
    joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    parents = [model.body_parentid[i] - 1 if model.body_parentid[i] > 0 else -1
               for i in range(1, model.nbody)]

    offsets = []
    for i in range(1, model.nbody):
        mj_pos = model.body_pos[i]
        offsets.append(np.array([mj_pos[0] * 100, mj_pos[2] * 100, -mj_pos[1] * 100]))

    bvh_poss = np.zeros((qpos_array.shape[0], num_joints, 3))
    bvh_rots = np.zeros((qpos_array.shape[0], num_joints, 3))

    for f in range(qpos_array.shape[0]):
        q = qpos_array[f]

        # root position
        bvh_poss[f, 0] = [q[0] * 100, q[2] * 100, -q[1] * 100]

        # root quaternion
        mj_w, mj_vx, mj_vy, mj_vz = q[3], q[4], q[5], q[6]
        bvh_quat_xyzw = [mj_vx, mj_vz, -mj_vy, mj_w]
        bvh_rots[f, 0] = R.from_quat(bvh_quat_xyzw).as_euler('ZYX', degrees=True)

        # each subsequent joint: ball joint, 4 values in wxyz order
        q_idx = 7
        for j in range(1, num_joints):
            mj_w, mj_vx, mj_vy, mj_vz = q[q_idx], q[q_idx + 1], q[q_idx + 2], q[q_idx + 3]
            bvh_quat_xyzw = [mj_vx, mj_vz, -mj_vy, mj_w]
            bvh_rots[f, j] = R.from_quat(bvh_quat_xyzw).as_euler('ZYX', degrees=True)
            q_idx += 4

    return {
        'rotations': bvh_rots,
        'positions': bvh_poss,
        'offsets': offsets,
        'parents': parents,
        'names': joint_names,
        'order': 'zyx',
        'frametime': frame_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retarget Flomo Xsens BVH to Geno/LAFAN1 skeleton BVH")
    parser.add_argument("--bvh_file", required=True, type=str, help="Input Flomo Xsens BVH file")
    parser.add_argument("--output_bvh", required=True, type=str, help="Output Geno/LAFAN1 BVH file")
    parser.add_argument("--skip_tpose", action="store_true", help="Skip first frame (T-pose)")
    args = parser.parse_args()

    frames, human_height, frame_time = load_flomo_bvh(
        args.bvh_file, skip_first_frame=args.skip_tpose
    )

    retargeter = GMR(src_human="bvh_flomo", tgt_robot="geno", actual_human_height=human_height)
    model = retargeter.model

    # --- joint groups ---
    # "preset" joints: source rotations set BEFORE IK. gives the solver a good
    # starting point so knees don't bend sideways.
    # "override" joints: source rotations set AFTER IK, completely replacing the
    # solver's output. we do this for arms, head, and feet because IK only needs
    # to solve root position, spine orientation, and foot placement — the rest
    # is better served by directly copying from source.
    preset_joints = ['LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg']
    override_joints = [
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
        'Neck', 'Head',
        'LeftFoot', 'RightFoot',
    ]
    all_copy_joints = preset_joints + override_joints

    # look up qpos addresses and parent body names for all copied joints
    copy_qpos_addr = {}
    copy_parent = {}
    for name in all_copy_joints:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        joint_id = model.body_jntadr[body_id]
        copy_qpos_addr[name] = model.jnt_qposadr[joint_id]
        parent_id = model.body_parentid[body_id]
        copy_parent[name] = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, parent_id)

    # --- twist offsets ---
    twist_offsets, twist_parent = compute_twist_offsets(model, frames[0], override_joints)
    print("Twist offsets (degrees):")
    for name in override_joints:
        angle = twist_offsets[name].magnitude() * 180 / np.pi
        print(f"  {name}: {angle:.1f}")

    # --- per-frame retargeting ---
    qpos_list = []
    print(f"Retargeting {len(frames)} frames...")
    for i, frame_data in enumerate(frames):
        # preset: seed leg rotations from source before IK
        for name in preset_joints:
            q = source_local_quat(model, frame_data, name, copy_parent)
            if q is not None:
                addr = copy_qpos_addr[name]
                retargeter.configuration.data.qpos[addr:addr + 4] = q
        mj.mj_forward(model, retargeter.configuration.data)

        # IK solve for root, spine, legs, feet positions
        qpos = retargeter.retarget(frame_data).copy()

        # override: copy arm/head/foot rotations with twist correction
        for name in override_joints:
            q_wxyz = source_local_quat(model, frame_data, name, copy_parent)
            if q_wxyz is not None:
                q_src = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                q_corrected = twist_parent[name].inv() * q_src * twist_offsets[name]
                xyzw = q_corrected.as_quat()
                addr = copy_qpos_addr[name]
                qpos[addr:addr + 4] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

        qpos_list.append(qpos)
        if (i + 1) % 100 == 0:
            print(f"  frame {i + 1}/{len(frames)}")

    # --- convert qpos to BVH and save ---
    qpos_array = np.array(qpos_list)
    bvh_data = qpos_to_bvh(model, qpos_array, frame_time)
    bvh_save(args.output_bvh, bvh_data)
    num_joints = model.nbody - 1
    print(f"Saved retargeted BVH to {args.output_bvh}")
    print(f"  {num_joints} joints, {qpos_array.shape[0]} frames, {1.0 / frame_time:.1f} fps")
