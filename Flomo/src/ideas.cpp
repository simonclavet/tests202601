#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cassert>

// Constants
static constexpr float ANIMATION_DT = 1.0f / 60.0f;

// Simple pose representation - just a vector of floats
using Pose = std::vector<float>;

// Animation data structure with precomputed velocities
struct Animation {
    std::vector<Pose> poses;      // Poses sampled at 60Hz
    std::vector<Pose> velocities; // Constant velocities between poses (per second)

    Animation() = default;

    explicit Animation(const std::vector<Pose>& poses)
        : poses(poses) {
        assert(!poses.empty() && "Animation must have at least one pose");
        assert(poses.size() >= 2 && "Animation must have at least 2 frames for velocities");
        precomputeVelocities();
    }

    // Precompute velocities: v[i] = (p[i+1] - p[i]) / dt
    void precomputeVelocities() {
        if (poses.empty()) return;

        size_t poseSize = poses[0].size();
        velocities.resize(poses.size()); // Same size as poses

        for (size_t i = 0; i < poses.size(); ++i) {
            velocities[i].resize(poseSize, 0.0f);

            if (i < poses.size() - 1) {
                for (size_t j = 0; j < poseSize; ++j) {
                    velocities[i][j] = (poses[i + 1][j] - poses[i][j]) / ANIMATION_DT;
                }
            }
            // Last frame has zero velocity
        }
    }

    // Get duration in seconds: (framecount - 1) * dt
    float getDuration() const {
        return (poses.size() - 1) * ANIMATION_DT;
    }

    // Get pose at specific time with linear interpolation
    Pose getPoseAtTime(float time) const {
        assert(time >= 0.0f && time <= getDuration() &&
            "Time must be within animation duration");

        if (poses.empty()) return {};

        // Direct frame lookup with interpolation
        float frameIndex = time / ANIMATION_DT;
        int frame0 = static_cast<int>(frameIndex);
        int frame1 = frame0 + 1;
        float alpha = frameIndex - frame0;

        // Handle boundary case
        if (frame0 >= static_cast<int>(poses.size()) - 1) {
            return poses.back();
        }

        // Clamp frame1
        frame1 = std::min(frame1, static_cast<int>(poses.size()) - 1);

        // Linear interpolation
        const Pose& p0 = poses[frame0];
        const Pose& p1 = poses[frame1];

        if (p0.size() != p1.size()) return p0;

        Pose result(p0.size());
        for (size_t i = 0; i < p0.size(); ++i) {
            result[i] = p0[i] + alpha * (p1[i] - p0[i]);
        }

        return result;
    }

    // Get velocity at specific time (constant between frames)
    Pose getVelocityAtTime(float time) const {
        assert(time >= 0.0f && time <= getDuration() &&
            "Time must be within animation duration");

        if (velocities.empty()) {
            return std::vector<float>(poses[0].size(), 0.0f);
        }

        // Find which frame interval we're in
        int frameIndex = static_cast<int>(time / ANIMATION_DT);
        frameIndex = std::min(frameIndex, static_cast<int>(velocities.size()) - 1);
        frameIndex = std::max(frameIndex, 0);

        return velocities[frameIndex];
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

// Main animation blending system
class AnimationBlender {
private:
    static constexpr int MAX_CURSORS = 5;  // Fixed pool size

    std::unordered_map<int, Animation> animations;
    AnimationCursor cursorPool[MAX_CURSORS];

    // State storage
    Pose currentPose;           // Current blended pose (output)
    Pose velBlendedPose;        // Velocity-blended pose
    Pose velBlendedVel;         // Velocity of velBlendedPose

    // Temporary storage
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
            else if (cursorPool[i].weight < minWeight) {
                minWeight = cursorPool[i].weight;
                smallestWeight = &cursorPool[i];
            }
        }

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
            result = pose;
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] *= weight;
            }
        }
        else {
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
        assert(anim.poses.size() >= 2 && "Animation must have at least 2 frames");
        animations[id] = anim;
    }

    // Remove animation from the library
    void removeAnimation(int id) {
        animations.erase(id);
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

        assert(startTime >= 0.0f && startTime <= animIt->second.getDuration() &&
            "Start time must be within animation duration");

        // Fade out existing cursors
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive) {
                cursorPool[i].setTargetWeight(0.0f);
            }
        }

        // Check if already playing
        auto* existingCursor = findCursor(animId);
        if (existingCursor) {
            existingCursor->currentTime = startTime;
            existingCursor->setTargetWeight(1.0f, true);
        }
        else {
            // Find available cursor
            AnimationCursor* cursor = findAvailableCursor();
            if (cursor) {
                cursor->reset();
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

    // Update all active animations
    void update(float currentTime) {
        const float deltaTime = previousUpdateTime > 0.0f ?
            currentTime - previousUpdateTime : 0.0f;
        previousUpdateTime = currentTime;

        if (deltaTime <= 0.0f) return;

        // Check if any active cursors
        bool anyActive = false;
        for (int i = 0; i < MAX_CURSORS; ++i) {
            if (cursorPool[i].isActive && cursorPool[i].weight > 0.001f) {
                anyActive = true;
                break;
            }
        }

        if (!anyActive) {
            if (poseInitialized) return;
            if (!currentPose.empty()) {
                std::fill(currentPose.begin(), currentPose.end(), 0.0f);
                poseInitialized = true;
            }
            return;
        }

        // Update cursors and accumulate
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

            // Update playback time
            cursorPool[i].currentTime += deltaTime;
            const float animDuration = animIt->second.getDuration();
            if (cursorPool[i].currentTime > animDuration) {
                cursorPool[i].currentTime = animDuration;
            }

            // Update blend weight
            cursorPool[i].updateWeight(deltaTime);
            const float weight = cursorPool[i].weight;

            // Get pose and accumulate
            const Pose& cursorPose = animIt->second.getPoseAtTime(cursorPool[i].currentTime);
            blendPosesInPlace(normalBlendedPose, totalWeight, cursorPose, weight);

            // Get velocity and accumulate
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

        // Clean up
        cleanupCursors();

        if (totalWeight < 0.0001f || normalBlendedPose.empty()) {
            return;
        }

        // Normalize
        normalizePose(normalBlendedPose, totalWeight);

        if (!blendedVel.empty() && totalWeight > 0.0001f) {
            const float invWeight = 1.0f / totalWeight;
            for (size_t i = 0; i < blendedVel.size(); ++i) {
                blendedVel[i] *= invWeight;
            }
        }

        // Initialize if needed
        if (!poseInitialized) {
            velBlendedPose = normalBlendedPose;
            velBlendedVel.resize(normalBlendedPose.size(), 0.0f);
            currentPose = normalBlendedPose;
            poseInitialized = true;
        }
        else {
            // Ensure correct size
            if (velBlendedPose.size() != normalBlendedPose.size()) {
                velBlendedPose.resize(normalBlendedPose.size(), 0.0f);
                velBlendedVel.resize(normalBlendedPose.size(), 0.0f);
            }

            // Step 1: Set velocity to blended velocity
            if (!blendedVel.empty() && blendedVel.size() == velBlendedVel.size()) {
                velBlendedVel = blendedVel;
            }

            // Step 2: Advance position with velocity
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

            // Output
            currentPose = velBlendedPose;
        }
    }

    // Get the current blended pose
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

    // Clear all active animations
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






