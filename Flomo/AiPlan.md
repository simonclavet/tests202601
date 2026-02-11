# AI Plan - Feb 11

## Done: BlendCursor carries its own poseGenFeatures segment

Goal: decouple cursor playback from AnimDatabase so a cursor only needs a short
segment of poseGenFeatures. This makes it easy to later replace the db lookup
with neural network output.

### What changed:
- AnimDatabase: added `poseGenFeaturesSegmentLength` (0.4s)
- BlendCursor: added `segment` (Array2D<float>), `segmentFrameTime`. Renamed `animTime` to `segmentAnimTime` (relative to segment start, not clip start)
- SpawnBlendCursor: now takes `const AnimDatabase* db`, copies a segment of poseGenFeatures rows into the cursor, resets segmentAnimTime to 0
- Per-cursor sampling loop: commented out old db-based sampling. Now deserializes PoseFeatures from two adjacent segment rows and interpolates. Fills both current and lookahead fields identically (lookahead IS the primary data now)
- Time advancement: clamps to segment length instead of clip length
- Everything downstream (blending, lookahead dragging, toe IK) untouched

### What's NOT changed (kept for comparison):
- All old BlendCursor fields still exist (localPositions, lookaheadRotations6d, etc)
- Old sampling code in comments
- Basic blend mode still in code
- AnimDatabase still holds all its arrays (we just don't read them during cursor playback anymore, except for IK indices and feature config)

## Done: Segment Autoencoder for poseGenFeatures

Goal: compress/reconstruct flattened poseGenFeatures segments so we have a latent
space a future predictor network can target.

### What changed:
- AnimDatabase: added normalization stats (poseGenFeaturesMean, poseGenFeaturesStd, poseGenFeaturesWeight), normalizedPoseGenFeatures, poseGenSegmentFrameCount, poseGenSegmentFlatDim
- Normalization in AnimDatabaseRebuild: per-dim mean/std, per-dim bone weight from GetBoneWeight table, normalized = (raw - mean) / std * boneWeight
- NetworkState: added segmentAutoEncoder, segmentOptimizer, isTrainingSegmentAE, segmentAELoss, segmentAEIterations
- networks.h: NetworkInitSegmentAutoEncoder (flatDim->512->256->128->256->512->flatDim), NetworkTrainSegmentAutoEncoder (samples random segments from clips, DAE with gaussian noise), save/load
- flomo.cpp: UI buttons for segment AE training, per-frame training call, save/load wired up
