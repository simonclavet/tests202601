# Structured Feature Augmentation for Training

## Context

GlimpseFlow and GlimpseDecompressor are conditioned on normalized MM features. The future part of these features (future velocity, future speed, aim direction) represents what gameplay *wants* to happen. Small deviations from the exact feature values should produce similar motion — aiming 10 degrees to the left, or moving 10% faster, shouldn't wildly change the output.

By adding **structured noise** during training (not white noise, but semantically meaningful perturbations like "scale all future speeds by 0.9"), we get implicit data augmentation without actually duplicating data. This should make the networks more robust to slight feature variations.

**Starting with speed only** — scale all future velocity vectors by a random factor per training sample.

## Design Constraints

- **ONE function** for computing features for a single frame. Called from the rebuild loop AND from training. No copy-pasting a big chunk.
- The rebuild loop calls it with `doAugmentation = false`, training calls it with `doAugmentation = true`.
- Feature names/types population: controlled by a `bool populateNames` argument (true only on first frame during rebuild).
- Each feature type handles its own augmentation internally when the flag is set. This way future augmentations (e.g. global timescale that would scale everything in weird ways) are handled locally per feature.
- We already have the motion_matching.h runtime query as a separate code path that must stay in sync. We don't want a third copy. This function is the ONE database-side implementation.

## Plan

### Step 1: Create the single-frame feature function

**File: `src/anim_database.h`**

```cpp
// Compute raw (un-normalized) features for a single frame.
// Writes db->featureDim floats into outFeatures.
// If populateNames is true, pushes feature names and types into db.
// If doAugmentation is true, each feature applies its own structured noise.
static inline void ComputeRawFeaturesForFrame(
    AnimDatabase* db,
    const MotionMatchingFeaturesConfig& cfg,
    int frame,
    float* outFeatures,
    bool populateNames,
    bool doAugmentation);
```

The function body is the current loop body (lines ~1174-1692 of anim_database.h), extracted as-is. The `isFirstFrame` name-push blocks become `if (populateNames) { ... }`.

**Augmentation for speed** (first pass, starting simple):
- When `doAugmentation` is true, at the top of the function, sample a random speed scale: `float speedScale = 1.0f + RandomFloat11() * 0.2f;` (range [0.8, 1.2])
- In **FutureVel**, **FutureVelClamped**, **FutureSpeed** blocks: multiply `futureVelAnimSpace` by `speedScale` before using it
- All other features: unchanged for now, but each block has a natural place to add its own augmentation later (e.g. rotate aim direction, scale toe velocities, etc.)

### Step 2: Refactor the rebuild loop
Look at "F:\experiments\tests202601\Flomo\CLAUDE.md" in for my coding style. Today I want you to help me use the        code in F:\experiments\GMR to retarget animations made with xsens such as                                               "F:\experiments\tests202601\Flomo\data\timi\xs_20251101_aleblanc_lantern_nav-002.fbx.bvh" to the lafan1 skeleton:        "F:\experiments\tests202601\Flomo\data\lafan\bvh\aiming2_subject2.bvh". Investigate in detail how to use GMR.          And how to extend it correctly. I also need to install its dependencies and do the python, conda, dancing around.        My miniconda is in C:\Users\LocalUser\miniconda3 so you might be forced to use the python from the environment         we will create. You should have access to the conda command, but to execute python you will need to find the            correct python from the environment. Your goal is to make me very good at retargetting animations from anything         to anything, using state of the art retargetters. Ultimatelly I would also like to port all GMR to pure c++.            Study carefully how GMR works, make me a tutorial, make a big plan. 