#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cassert>

// Constants
static constexpr float ANIMATION_DT = 1.0f / 60.0f;

// Simple pose representation - just a vector of floats
using Pose = std::vector<float>;

// Animation data structure with precomputed velocities at mid-points
struct Animation {
    std::vector<Pose> poses;      // Poses sampled at 60Hz
    std::vector<Pose> velocities; // Velocities at mid-points between poses

    Animation() = default;

    explicit Animation(const std::vector<Pose>& poses)
        : poses(poses) {
        assert(!poses.empty() && "Animation must have at least one pose");
        assert(poses.size() >= 2 && "Animation must have at least 2 frames for velocities");
        precomputeMidPointVelocities();
    }

    // Precompute velocities at mid-points: v_mid[i] = p[i+1] - p[i]
    void precomputeMidPointVelocities() {
        if (poses.empty()) return;

        size_t poseSize = poses[0].size();
        velocities.resize(poses.size() - 1); // One less than poses

        for (size_t i = 0; i < poses.size() - 1; ++i) {
            velocities[i].resize(poseSize, 0.0f);
            for (size_t j = 0; j < poseSize; ++j) {
                velocities[i][j] = poses[i + 1][j] - poses[i][j];
            }
        }
    }

    // Get duration in seconds: (framecount - 1) * dt
    float getDuration() const {
        return (poses.size() - 1) * ANIMATION_DT;
    }

    // Get pose at specific time with interpolation
    Pose getPoseAtTime(float time) const {
        // Assert time is within bounds
        assert(time >= 0.0f && time <= getDuration() &&
            "Time must be within animation duration");

        if (poses.empty()) return {};
        if (poses.size() == 1) return poses[0];

        // Calculate frame indices for interpolation
        float frameIndex = time / ANIMATION_DT;
        int frame0 = static_cast<int>(frameIndex);
        int frame1 = frame0 + 1;
        float alpha = frameIndex - frame0;

        // Handle boundary cases
        if (frame0 >= static_cast<int>(poses.size()) - 1) {
            return poses.back();
        }

        // Ensure frame1 doesn't exceed bounds
        frame1 = std::min(frame1, static_cast<int>(poses.size()) - 1);

        // Linear interpolation between frames
        const Pose& pose0 = poses[frame0];
        const Pose& pose1 = poses[frame1];

        if (pose0.size() != pose1.size()) {
            return pose0;
        }

        Pose result(pose0.size());
        for (size_t i = 0; i < pose0.size(); ++i) {
            result[i] = pose0[i] + alpha * (pose1[i] - pose0[i]);
        }

        return result;
    }

    // Get velocity at specific time with mid-point interpolation
    Pose getVelocityAtTime(float time) const {
        // Assert time is within bounds
        assert(time >= 0.0f && time <= getDuration() &&
            "Time must be within animation duration");

        if (velocities.empty()) {
            return std::vector<float>(poses[0].size(), 0.0f);
        }

        const size_t poseSize = poses[0].size();

        // Special cases for boundaries
        if (time <= ANIMATION_DT * 0.5f) {
            // At beginning, use first mid-point velocity
            return velocities[0];
        }

        if (time >= getDuration() - ANIMATION_DT * 0.5f) {
            // At end, use last mid-point velocity
            return velocities.back();
        }

        // For interior times: find the two surrounding mid-points and interpolate
        // Mid-point i is at time (i + 0.5) * dt
        float midPointTime = time - ANIMATION_DT * 0.5f;
        int midPointIndex = static_cast<int>(midPointTime / ANIMATION_DT);

        // Clamp index
        midPointIndex = std::max(0, std::min(midPointIndex, static_cast<int>(velocities.size()) - 2));

        // Time relative to first mid-point
        float t0 = (midPointIndex + 0.5f) * ANIMATION_DT;
        float alpha = (time - t0) / ANIMATION_DT;

        // Ensure alpha is in [0, 1]
        alpha = std::max(0.0f, std::min(1.0f, alpha));

        const Pose& vel0 = velocities[midPointIndex];
        const Pose& vel1 = velocities[midPointIndex + 1];

        if (vel0.size() != poseSize || vel1.size() != poseSize) {
            return std::vector<float>(poseSize, 0.0f);
        }

        // Linear interpolation between mid-point velocities
        Pose result(poseSize);
        for (size_t i = 0; i < poseSize; ++i) {
            result[i] = vel0[i] + alpha * (vel1[i] - vel0[i]);
        }

        return result;
    }
};

// Critically damped spring implementation (optimized)
class CriticallyDampedSpring {
private:
    float k;       // Stiffness
    float d;       // Damping
    float vel;     // Current velocity

public:
    CriticallyDampedSpring(float stiffness = 10.0f)
        : k(stiffness), d(2.0f * std::sqrt(stiffness)), vel(0.0f) {}

    // Fast update
    float update(float current, float target, float deltaTime) {
        if (deltaTime <= 0.0f) return current;

        const float x = current - target;

        // Early out if very close
        if (x * x < 1e-6f && vel * vel < 1e-6f) {
            return target;
        }

        // Simplified spring update
        const float acceleration = -k * x - d * vel;
        vel += acceleration * deltaTime;
        current += vel * deltaTime;

        return current;
    }

    void resetVelocity() { vel = 0.0f; }
    float getVelocity() const { return vel; }
};

// Animation cursor
struct AnimationCursor {
    int animId;           // Reference to animation data
    float currentTime;    // Current playback time in seconds
    float weight;         // Current blending weight (0-1)
    float targetWeight;   // Target blending weight (0-1)
    CriticallyDampedSpring weightSpring;  // Spring for weight transitions
    bool isActive;        // Whether this cursor is in use

    AnimationCursor()
        : animId(-1), currentTime(0.0f), weight(0.0f), targetWeight(0.0f),
        weightSpring(30.0f), isActive(false) {}

    void initialize(int id, float startTime, float target, float stiffness) {
        animId = id;
        currentTime = startTime;
        weight = 0.0f;
        targetWeight = target;
        weightSpring = CriticallyDampedSpring(stiffness);
        isActive = true;
    }

    void reset() {
        animId = -1;
        weight = 0.0f;
        targetWeight = 0.0f;
        isActive = false;
    }

    // Update weight using spring
    void updateWeight(float deltaTime) {
        weight = weightSpring.update(weight, targetWeight, deltaTime);
    }

    // Set target weight and reset spring if needed
    void setTargetWeight(float target, bool resetSpring = false) {
        targetWeight = std::max(0.0f, std::min(target, 1.0f));
        if (resetSpring) {
            weightSpring.resetVelocity();
            weight = targetWeight;
        }
    }

    // Check if this cursor can be removed
    bool canRemove() const {
        return !isActive || (weight <= 0.0001f && targetWeight <= 0.0001f);
    }
};

// Main animation blending system with improved precision velocity integration
class AnimationBlender {
private:
    static constexpr int MAX_CURSORS = 5;  // Fixed pool size

    std::unordered_map<int, Animation> animations;
    AnimationCursor cursorPool[MAX_CURSORS];  // Fixed array for cursor pool

    // State storage
    Pose currentPose;           // Current blended pose (output)
    Pose velBlendedPose;        // Velocity-blended pose
    Pose velBlendedVel;         // Velocity of velBlendedPose

    // Temporary storage reused across frames
    Pose normalBlendedPose;
    Pose blendedVel;

    float previousUpdateTime;
    float blendTime;           // Time for blending between animations
    float posStiffness;        // Stiffness for position correction
    bool poseInitialized;

    // Find inactive cursor or one with smallest weight
    AnimationCursor* findAvailableCursor() {
        AnimationCursor* available = nullptr;
        AnimationCursor* smallestWeight = nullptr;
        float minWeight = 1e9f;

        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (!cursorPool[i].isActive) {
                available = &cursorPool[i];
                break;
            }
            else {
                if (cursorPool[i].weight < minWeight) {
                    minWeight = cursorPool[i].weight;
                    smallestWeight = &cursorPool[i];
                }
            }
        }

        // If no inactive cursor, recycle one with smallest weight
        return available ? available : smallestWeight;
    }

    // Find cursor by animId
    AnimationCursor* findCursor(int animId) {
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].animId == animId) {
                return &cursorPool[i];
            }
        }
        return nullptr;
    }

    // Fast pose blending without allocation
    void blendPosesInPlace(Pose& result, float& totalWeight,
        const Pose& pose, float weight) {
        if (weight <= 0.0001f || pose.empty()) return;

        if (totalWeight == 0.0f) {
            // First pose with weight
            if (result.size() != pose.size()) {
                result.resize(pose.size());
            }
            for (size_t i = 0; i < pose.size(); ++i) {
                result[i] = pose[i] * weight;
            }
        }
        else {
            // Accumulate weighted pose
            if (result.size() != pose.size()) {
                result.resize(pose.size(), 0.0f);
            }
            for (size_t i = 0; i < pose.size(); ++i) {
                result[i] += pose[i] * weight;
            }
        }

        totalWeight += weight;
    }

    // Fast normalization
    void normalizePose(Pose& pose, float totalWeight) {
        if (totalWeight > 0.0001f) {
            const float invTotalWeight = 1.0f / totalWeight;
            for (size_t i = 0; i < pose.size(); ++i) {
                pose[i] *= invTotalWeight;
            }
        }
    }

    // Remove inactive cursors
    void cleanupCursors() {
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].canRemove()) {
                cursorPool[i].reset();
            }
        }
    }

public:
    AnimationBlender(float blendDuration = 0.2f,
        float posStiffnessParam = 5.0f)
        : previousUpdateTime(0.0f),
        blendTime(blendDuration),
        posStiffness(posStiffnessParam),
        poseInitialized(false) {
        // Initialize cursor pool
        for (int i = 0; i < MAX_CURSORS; ++i) {
            cursorPool[i] = AnimationCursor();
        }
    }

    // Add animation to the library
    void addAnimation(int id, const Animation& anim) {
        // Verify animation has at least 2 frames
        assert(anim.poses.size() >= 2 && "Animation must have at least 2 frames");
        animations[id] = anim;
    }

    // Remove animation from the library
    void removeAnimation(int id) {
        animations.erase(id);
        // Deactivate any cursors using this animation
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].animId == id) {
                cursorPool[i].reset();
            }
        }
    }

    // Start a new animation
    void startAnim(int animId, float startTime = 0.0f) {
        auto animIt = animations.find(animId);
        if (animIt == animations.end()) return;

        // Assert start time is within animation bounds
        assert(startTime >= 0.0f && startTime <= animIt->second.getDuration() &&
            "Start time must be within animation duration");

        // Set all existing cursors to fade out
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive) {
                cursorPool[i].setTargetWeight(0.0f);
            }
        }

        // Check if this animation is already playing
        auto* existingCursor = findCursor(animId);
        if (existingCursor) {
            existingCursor->currentTime = startTime;
            existingCursor->setTargetWeight(1.0f, true);
        }
        else {
            // Find available cursor from pool
            AnimationCursor* cursor = findAvailableCursor();
            if (cursor) {
                cursor->reset();  // Reset if recycling
                cursor->initialize(animId, startTime, 1.0f, 1.0f / (blendTime * blendTime));

                // Start at full weight if no other active animations
                bool anyActive = false;
                for (int i = 0; i < MAX_CURSORS; ++i) {
                    if (cursorPool[i].isActive && cursorPool[i].weight > 0.001f) {
                        anyActive = true;
                        break;
                    }
                }

                if (!anyActive) {
                    cursor->weight = 1.0f;
                    cursor->weightSpring.resetVelocity();
                }
            }
        }
    }

    // Update all active animations and compute blended pose with improved precision
    void update(float currentTime) {
        const float deltaTime = previousUpdateTime > 0.0f ?
            currentTime - previousUpdateTime : 0.0f;
        previousUpdateTime = currentTime;

        if (deltaTime <= 0.0f) return;

        // Early out if no active cursors
        bool anyActive = false;
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].weight > 0.001f) {
                anyActive = true;
                break;
            }
        }

        if (!anyActive) {
            if (poseInitialized) {
                // Keep current pose
                return;
            }
            else if (!currentPose.empty()) {
                // Initialize with zeros
                std::fill(currentPose.begin(), currentPose.end(), 0.0f);
                poseInitialized = true;
            }
            return;
        }

        // First pass: advance all cursor times by half deltaTime for mid-point sampling
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (!cursorPool[i].isActive) continue;

            cursorPool[i].currentTime += deltaTime * 0.5f;

            auto animIt = animations.find(cursorPool[i].animId);
            if (animIt != animations.end()) {
                const float animDuration = animIt->second.getDuration();
                if (cursorPool[i].currentTime > animDuration) {
                    cursorPool[i].currentTime = animDuration;
                }
            }
        }

        // Compute blended pose and velocity at the mid-point
        float totalWeight = 0.0f;
        normalBlendedPose.clear();
        blendedVel.clear();

        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (!cursorPool[i].isActive || cursorPool[i].weight <= 0.001f) {
                continue;
            }

            auto animIt = animations.find(cursorPool[i].animId);
            if (animIt == animations.end()) {
                continue;
            }

            // Update blend weight
            cursorPool[i].updateWeight(deltaTime);
            const float weight = cursorPool[i].weight;

            // Get pose at mid-point time (with interpolation)
            const Pose& cursorPose = animIt->second.getPoseAtTime(cursorPool[i].currentTime);
            blendPosesInPlace(normalBlendedPose, totalWeight, cursorPose, weight);

            // Get velocity at mid-point time (with mid-point interpolation)
            const Pose& cursorVel = animIt->second.getVelocityAtTime(cursorPool[i].currentTime);
            if (!cursorVel.empty()) {
                if (blendedVel.empty()) {
                    blendedVel.resize(cursorVel.size(), 0.0f);
                }

                if (blendedVel.size() == cursorVel.size()) {
                    for (size_t j = 0; j < cursorVel.size(); ++j) {
                        blendedVel[j] += cursorVel[j] * weight;
                    }
                }
            }
        }

        // Clean up inactive cursors
        cleanupCursors();

        // If no valid poses, return
        if (totalWeight < 0.0001f || normalBlendedPose.empty()) {
            return;
        }

        // Normalize normal blended pose
        normalizePose(normalBlendedPose, totalWeight);

        // Normalize blended velocity
        if (!blendedVel.empty() && totalWeight > 0.0001f) {
            const float invWeight = 1.0f / totalWeight;
            for (size_t i = 0; i < blendedVel.size(); ++i) {
                blendedVel[i] *= invWeight;
            }
        }

        // Initialize velocity-blended state if needed
        if (!poseInitialized) {
            velBlendedPose = normalBlendedPose;
            velBlendedVel.resize(normalBlendedPose.size(), 0.0f);
            currentPose = normalBlendedPose;
            poseInitialized = true;
        }
        else {
            // Ensure velBlendedPose has correct size
            if (velBlendedPose.size() != normalBlendedPose.size()) {
                velBlendedPose.resize(normalBlendedPose.size(), 0.0f);
                velBlendedVel.resize(normalBlendedPose.size(), 0.0f);
            }

            // Step 1: Set velocity to blended mid-point velocity
            if (!blendedVel.empty() && blendedVel.size() == velBlendedVel.size()) {
                velBlendedVel = blendedVel;
            }

            // Step 2: Advance position with velocity using full deltaTime
            // (This uses velocity at mid-point, which gives second-order accuracy)
            for (size_t i = 0; i < velBlendedPose.size(); ++i) {
                velBlendedPose[i] += velBlendedVel[i] * deltaTime;
            }

            // Step 3: Lerp position toward normal blended pose
            const float posAlpha = posStiffness * deltaTime;
            if (posAlpha < 1.0f) {
                const float invPosAlpha = 1.0f - posAlpha;
                for (size_t i = 0; i < velBlendedPose.size(); ++i) {
                    velBlendedPose[i] = velBlendedPose[i] * invPosAlpha + normalBlendedPose[i] * posAlpha;
                }
            }
            else {
                velBlendedPose = normalBlendedPose;
            }

            // Update output pose
            currentPose = velBlendedPose;
        }

        // Second half of time advancement for cursors
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (!cursorPool[i].isActive) continue;

            cursorPool[i].currentTime += deltaTime * 0.5f;

            auto animIt = animations.find(cursorPool[i].animId);
            if (animIt != animations.end()) {
                const float animDuration = animIt->second.getDuration();
                if (cursorPool[i].currentTime > animDuration) {
                    cursorPool[i].currentTime = animDuration;
                }
            }
        }
    }

    // Get the current blended pose (computed in update)
    const Pose& getBlendedPose() const {
        return currentPose;
    }

    // Set stiffness parameter
    void setStiffness(float posStiffnessParam) {
        posStiffness = posStiffnessParam;
    }

    // Get stiffness parameter
    float getStiffness() const {
        return posStiffness;
    }

    // Get blend time
    float getBlendTime() const {
        return blendTime;
    }

    // Set blend time
    void setBlendTime(float blendDuration) {
        blendTime = blendDuration;
    }

    // Get number of active animations
    int getActiveAnimationCount() const {
        int count = 0;
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive) count++;
        }
        return count;
    }

    // Check if a specific animation is currently playing
    bool isAnimPlaying(int animId) const {
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].animId == animId &&
                cursorPool[i].weight > 0.001f) {
                return true;
            }
        }
        return false;
    }

    // Clear all active animations immediately
    void clearActiveAnimations() {
        for (int i = 0; i < MAX_CURSORS; ++i) {
            cursorPool[i].reset();
        }

        if (!currentPose.empty()) {
            std::fill(currentPose.begin(), currentPose.end(), 0.0f);
            std::fill(velBlendedPose.begin(), velBlendedPose.end(), 0.0f);
            std::fill(velBlendedVel.begin(), velBlendedVel.end(), 0.0f);
        }
    }

    // Reset pose system
    void resetPose() {
        currentPose.clear();
        velBlendedPose.clear();
        velBlendedVel.clear();
        normalBlendedPose.clear();
        blendedVel.clear();
        poseInitialized = false;
    }
};



/*

What I can see is that sometimes the error between basic and lookahead is large even if the anim is not extremely fast.
And error sometimes is mostly due to hips rotation. The guy is sometimes banked more in basic than in lookahead, and the 
legs and torso are significantly offseted in opposite direction (hips just rotated not enough)




*/