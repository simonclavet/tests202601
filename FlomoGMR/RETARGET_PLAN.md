# Plan: Xsens-to-LAFAN1 Motion Retargeting via GMR

## Context

**Goal:** Retarget Xsens motion capture BVH animations onto the LAFAN1/Geno skeleton, output as BVH. Eventually build a standalone retargeter and port to C++.

**Key discovery:** The Geno skeleton IS the LAFAN1 skeleton (77 joints, fingers included). GMR already has:
- `scripts/xsens_to_geno_bvh.py` - retargets Xsens BVH -> Geno skeleton -> BVH output
- `bvh_xsens_to_geno.json` - IK config mapping Xsens joints to Geno bodies
- `assets/geno/geno.xml` - MuJoCo XML generated from `Geno_bind.bvh`
- `generate_geno_xml.py` - converts LAFAN1-style BVH to MuJoCo XML

**Problem:** The existing Xsens loader (`load_xsens_file()`) expects raw Xsens BVH names (Pelvis, L5, T8, LeftHip, LeftAnkle) and creates synthetic `LeftFootMod`/`RightFootMod` joints. Our Flomo-converted Xsens BVH uses standard BVH names (Hips, Spine, LeftUpLeg, LeftFoot) which match LAFAN1 naming. We need a loader that handles this naming convention.

**Source file:** `F:\experiments\tests202601\Flomo\data\timi\xs_20251101_aleblanc_lantern_nav-002.fbx.bvh`
- 17 joints, 30fps, positions in meters, Y-up, rotation order ZXY
- First frame is T-pose (useful for calibration)

**Target skeleton:** Geno/LAFAN1 (77 joints, 6 channels each, ZYX rotation, positions in cm)
- Bind pose: `F:\experiments\GenoView\resources\Geno_bind.bvh`
- Stance pose: `F:\experiments\GenoView\resources\Geno_stance.bvh`
- MuJoCo XML: `F:\experiments\GMR\assets\geno\geno.xml`

---

## Phase 0: Environment Setup

### 0.1 Create conda environment
```bash
conda create -n gmr python=3.10 -y
conda activate gmr
```

### 0.2 Install GMR
```bash
cd F:\experiments\GMR
pip install -e .
pip install daqp
```
Note: `pip install mujoco` (pulled by setup.py) includes precompiled binaries. The user also has precompiled MuJoCo at `F:\experiments\GMR\third_party\mujoco-3.5.0-windows-x86_64` if we need C headers later for the C++ port. Building MuJoCo from source is only needed for the C++ port phase.

### 0.3 Verify GMR runs
Test with the existing LAFAN1-to-Geno pipeline (no Xsens loader needed):
```bash
python scripts/bvh_to_robot.py --bvh_file "F:\experiments\tests202601\Flomo\data\lafan\bvh\aiming2_subject2.bvh" --robot geno --format lafan1
```
This should open a MuJoCo viewer showing the Geno skeleton performing the animation.

---

## Phase 1: Flomo Xsens -> Geno/LAFAN1 BVH Pipeline

### 1.1 Create a loader for Flomo-converted Xsens BVH

**File:** `F:\experiments\GMR\general_motion_retargeting\utils\flomo_bvh.py`

This loader handles BVH files converted by Flomo's fbx2bvh with standard BVH naming.

It needs to:
1. Parse the BVH file (reuse `lafan_vendor/extract.py::read_bvh()`)
2. Convert rotation order from ZXY to the internal format
3. Compute forward kinematics -> global positions + quaternions
4. Return per-frame dict: `{"Hips": (pos_3d, quat_wxyz), "Spine": (...), ...}`
5. Handle meters (no /100 scaling unlike LAFAN1 which is in cm)
6. Create synthetic `LeftFootMod`/`RightFootMod` if the IK config expects them, OR use the `bvh_xsens_to_geno.json` config which maps `LeftFoot`/`RightFoot` directly

Key reference files:
- `F:\experiments\GMR\general_motion_retargeting\utils\lafan1.py` (LAFAN1 loader pattern)
- `F:\experiments\GMR\general_motion_retargeting\utils\xsens.py` (Xsens loader pattern)
- `F:\experiments\GMR\general_motion_retargeting\utils\lafan_vendor\extract.py` (BVH parser)
- `F:\experiments\GMR\general_motion_retargeting\utils\lafan_vendor\utils.py` (quat_fk)

### 1.2 Register new source format in params.py

**File:** `F:\experiments\GMR\general_motion_retargeting\params.py`

Add to `IK_CONFIG_DICT`:
```python
"bvh_flomo": {
    "geno": IK_CONFIG_ROOT / "bvh_xsens_to_geno.json",  # reuse existing config
}
```

The existing `bvh_xsens_to_geno.json` already uses standard BVH names (Hips, LeftHand, LeftFoot, etc.) which match our Flomo BVH joint names. No new IK config needed.

### 1.3 Create retargeting script

**File:** `F:\experiments\GMR\scripts\flomo_to_geno_bvh.py`

Based on `F:\experiments\GMR\scripts\xsens_to_geno_bvh.py` (120 lines). Changes:
- Use new `load_flomo_bvh()` loader instead of `load_xsens_file()`
- Use `src_human="bvh_flomo"` and `tgt_robot="geno"`
- Keep the existing BVH output logic (coordinate transform: MJ(x,y,z) -> BVH(x*100, z*100, -y*100), ZYX euler output)
- Skip the T-pose first frame (or make it configurable with `--skip_tpose`)

### 1.4 Test and verify

Run the retargeting:
```bash
python scripts/flomo_to_geno_bvh.py \
  --bvh_file "F:\experiments\tests202601\Flomo\data\timi\xs_20251101_aleblanc_lantern_nav-002.fbx.bvh" \
  --output_bvh "F:\experiments\tests202601\output_lafan1.bvh"
```

Verify output by loading in Flomo:
```bash
F:\experiments\tests202601\Flomo\build\src\Debug\flomo.exe output_lafan1.bvh
```

### 1.5 Weight tuning (if needed)

If retargeting looks wrong, iterate on the IK config. Common fixes:
- Coordinate frame issues: check the Y-up -> Z-up conversion in the loader
- Scale issues: check human_height_assumption (1.75m) vs actual skeleton height
- Rotation offsets: add quaternion offsets in the IK config for misaligned joints

---

## Phase 2: Direct BVH-to-BVH Retargeter (No MuJoCo)

Build a standalone retargeter for the eventual C++ port.

### 2.1 Architecture

For topologically similar human skeletons, the algorithm is:
1. Parse source BVH -> local rotations + root position per frame
2. Map source joints to target joints by name
3. Copy local rotations directly (with optional rotation offset)
4. Scale root position by skeleton height ratio
5. Unmapped target joints keep identity/rest-pose rotation
6. Write target BVH

This is simpler than GMR's IK approach and sufficient when both skeletons are human with similar topology.

### 2.2 Components (all numpy-only, no MuJoCo/mink)

**File:** `F:\experiments\GMR\bvh_retarget.py` (single file module)

Reuse from existing codebase:
- BVH parsing: `lafan_vendor/extract.py::read_bvh()` or `GenoView/resources/bvh.py`
- Quaternion math: `lafan_vendor/utils.py` (quat_fk, quat_mul, euler_to_quat)
- BVH writing: `xsens_to_geno_bvh.py::bvh_save()`

New code:
- Joint name mapping logic
- Height ratio computation
- Frame-by-frame rotation copy with offset application

### 2.3 Optional: Two-bone IK for end-effectors

If rotation copying produces bad hand/foot positions due to proportion differences, add analytical two-bone IK for arms (shoulder->elbow->wrist) and legs (hip->knee->ankle). This handles limb length differences without a full iterative solver.

---

## Phase 3: Tutorial Document

Create a concise guide covering:
- How GMR's IK-based retargeting works (the algorithm in plain English)
- How to add a new source skeleton (loader + IK config)
- How to add a new target skeleton (MuJoCo XML + registration)
- How to tune IK weights and rotation offsets
- Coordinate system conventions (BVH Y-up vs MuJoCo Z-up)

---

## Phase 4: C++ Port (Later)

Architecture notes for when we get there:
- Reimplement the Phase 2 direct retargeter in C++
- Use Eigen for vector/quaternion math (or roll our own small math lib)
- Single-header BVH parser (text parsing, no library needed)
- Integrate with Flomo's existing `bvh_parser.h` and `transform_data.h`
- MuJoCo C API available at `F:\experiments\GMR\third_party\mujoco-3.5.0-windows-x86_64` if we want IK-based retargeting too
- No CUDA needed (retargeting is CPU-bound, per-frame is fast)

---

## Implementation Order

1. **Phase 0** - Environment setup (conda, pip install, verify)
2. **Phase 1.1** - Flomo BVH loader
3. **Phase 1.2** - Register in params.py
4. **Phase 1.3** - Retargeting script
5. **Phase 1.4** - Test and verify output
6. **Phase 1.5** - Tune weights if needed
7. **Phase 2** - Direct BVH-to-BVH retargeter (after Phase 1 works)
8. **Phase 3** - Tutorial
9. **Phase 4** - C++ port (separate effort)

---

## Files to Create

| File | Purpose |
|------|---------|
| `GMR/general_motion_retargeting/utils/flomo_bvh.py` | BVH loader for Flomo-converted files |
| `GMR/scripts/flomo_to_geno_bvh.py` | Main retargeting script |

## Files to Modify

| File | Change |
|------|--------|
| `GMR/general_motion_retargeting/params.py` | Add `bvh_flomo` source format |

## Key Reference Files

| File | Why |
|------|-----|
| `GMR/scripts/xsens_to_geno_bvh.py` | Template for our retargeting script |
| `GMR/general_motion_retargeting/utils/lafan1.py` | Loader pattern to follow |
| `GMR/general_motion_retargeting/utils/lafan_vendor/extract.py` | BVH parser we'll reuse |
| `GMR/general_motion_retargeting/utils/lafan_vendor/utils.py` | FK/quaternion math |
| `GMR/general_motion_retargeting/ik_configs/bvh_xsens_to_geno.json` | IK config (reuse as-is) |
| `GMR/general_motion_retargeting/motion_retarget.py` | Core GMR algorithm |
| `GMR/generate_geno_xml.py` | How Geno XML was generated |
| `GenoView/resources/Geno_bind.bvh` | Target skeleton bind pose |
