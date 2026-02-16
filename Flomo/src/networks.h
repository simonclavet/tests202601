#pragma once

#include "definitions.h"
#include <string>
#include <filesystem>
#include <unordered_map>
#include <chrono>

extern "C" void cuda_init_context();

constexpr int FEATURE_AE_LATENT_DIM = 16;
constexpr int SEGMENT_AE_LATENT_DIM = 128;
constexpr int POSE_AE_LATENT_DIM = 16;
constexpr double HEADSTART_SECONDS = 5.0;
constexpr double FRIDAY_FLOW_HEADSTART = 20.0;
constexpr double AUTOSAVE_INTERVAL_SECONDS = 300.0;
constexpr int FLOW_INFERENCE_STEPS = 8;
constexpr double LOSS_LOG_INTERVAL_SECONDS = 5.0;

// training time budget weights (relative, get normalized per phase)
constexpr float BUDGET_WEIGHT_FEATURE_AE = 1.0f;
constexpr float BUDGET_WEIGHT_SEGMENT_AE = 0.1f;
constexpr float BUDGET_WEIGHT_LATENT_SEGMENT_PREDICTOR = 0.1f;
constexpr float BUDGET_WEIGHT_SEGMENT_E2E = 0.1f;
constexpr float BUDGET_WEIGHT_POSE_E2E = 1.0f;
constexpr float BUDGET_WEIGHT_SEGMENT_FLOW = 0.1f;
constexpr float BUDGET_WEIGHT_POSE_AE = 1.0f;
constexpr float BUDGET_WEIGHT_FULL_FLOW = 0.1f;
constexpr float BUDGET_WEIGHT_FRIDAY_FLOW = 0.1f;
constexpr float BUDGET_WEIGHT_SINGLE_POSE_PREDICTOR = 3.0f;
constexpr float BUDGET_WEIGHT_UNCOND_ADVANCE = 0.5f;
constexpr float BUDGET_WEIGHT_MONDAY = 1.0f;

// switch between ReLU and GELU for all networks
#define USE_GELU
#ifdef USE_GELU
    #define NN_ACTIVATION() torch::nn::GELU()
    #define nn_activate(x) torch::gelu(x)
#else
    #define NN_ACTIVATION() NN_ACTIVATION()
    #define nn_activate(x) nn_activate(x)
#endif

// Flow matching mode: if defined, predict delta from average latent (condition on avgLatent)
// if undefined, predict segment latent directly (don't condition on avgLatent)
//#define DO_DELTA_FLOWMATCHING

// Flow matching parameterization: if defined, predict endpoint x1 directly
// if undefined, predict velocity field v = (x1 - xt) / (1 - t)
//#define FLOW_PREDICT_X1

constexpr float FRIDAY_FLOW_POSE_NOISE = 0.5f;  // noise added to z0 conditioning during training (robustness)
constexpr float POSE_AE_STRAIGHT_WEIGHT = 0.1f;  // weight for latent trajectory straightness loss
constexpr float POSE_AE_TIGHT_WEIGHT = 0.1f;     // weight for latent trajectory tightness loss (small steps)

constexpr float LOSS_SMOOTHING_FACTOR = 0.1f;  // lerp factor: new loss weight (0=fully smoothed, 1=no smoothing)

// pick a random legal frame using cluster-stratified sampling if available,
// otherwise fall back to plain uniform
static inline int SampleLegalFrame(const AnimDatabase* db)
{
    if (db->clusterCount <= 0)
        return db->legalStartFrames[RandomInt((int)db->legalStartFrames.size())];

    const int c = RandomInt(db->clusterCount);
    const std::vector<int>& frames = db->clusterFrames[c];
    return frames[RandomInt((int)frames.size())];
}

//---------------------------------------------------------
// encoder / decoder helpers
//---------------------------------------------------------

// Both AEs have the same layer layout:
// [0] Linear  [1] Activation  [2] Linear  [3] Activation  [4] Linear
// (encoder bottleneck)
// [5] Linear  [6] Activation  [7] Linear  [8] Activation  [9] Linear

// runs only the encoder half (layers 0-4)
static inline torch::Tensor EncodeWithAE(
    torch::nn::Sequential& ae,
    const torch::Tensor& input)
{
    std::vector<std::shared_ptr<torch::nn::Module>>
        children = ae->children();
    torch::nn::LinearImpl* l0 =
        children[0]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l2 =
        children[2]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l4 =
        children[4]->as<torch::nn::LinearImpl>();
    assert(l0 && l2 && l4);

    torch::Tensor x = input;
    x = nn_activate(l0->forward(x));
    x = nn_activate(l2->forward(x));
    x = l4->forward(x);
    return x;
}

// runs only the decoder half (layers 5-9)
static inline torch::Tensor DecodeWithAE(
    torch::nn::Sequential& ae,
    const torch::Tensor& latent)
{
    std::vector<std::shared_ptr<torch::nn::Module>>
        children = ae->children();
    torch::nn::LinearImpl* l5 =
        children[5]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l7 =
        children[7]->as<torch::nn::LinearImpl>();
    torch::nn::LinearImpl* l9 =
        children[9]->as<torch::nn::LinearImpl>();
    assert(l5 && l7 && l9);

    torch::Tensor x = latent;
    x = nn_activate(l5->forward(x));
    x = nn_activate(l7->forward(x));
    x = l9->forward(x);
    return x;
}

// PoseAE encoder: runs the first half of the sequential (layers 0..N/2-1)
static inline torch::Tensor PoseAEEncode(torch::nn::Sequential& ae, const torch::Tensor& input)
{
    std::vector<std::shared_ptr<torch::nn::Module>> children = ae->children();
    const int half = (int)children.size() / 2;
    torch::Tensor x = input;
    for (int i = 0; i < half; ++i)
    {
        torch::nn::LinearImpl* lin = children[i]->as<torch::nn::LinearImpl>();
        if (lin)
        {
            x = (i < half - 1) ? nn_activate(lin->forward(x)) : lin->forward(x);
        }
    }
    return x;
}

// PoseAE decoder: runs the second half of the sequential (layers N/2..N-1)
static inline torch::Tensor PoseAEDecode(torch::nn::Sequential& ae, const torch::Tensor& latent)
{
    std::vector<std::shared_ptr<torch::nn::Module>> children = ae->children();
    const int n = (int)children.size();
    const int half = n / 2;
    torch::Tensor x = latent;
    for (int i = half; i < n; ++i)
    {
        torch::nn::LinearImpl* lin = children[i]->as<torch::nn::LinearImpl>();
        if (lin)
        {
            x = (i < n - 1) ? nn_activate(lin->forward(x)) : lin->forward(x);
        }
    }
    return x;
}

//---------------------------------------------------------
// CUDA device setup (shared by all networks)
//---------------------------------------------------------

static inline void NetworkEnsureDevice(NetworkState* state)
{
    if (state->device.is_cuda()) return;

    TraceLog(LOG_INFO, "Initializing CUDA context...");
    cuda_init_context();

    const bool cudaOk = torch::cuda::is_available();
    state->device = cudaOk ? torch::kCUDA : torch::kCPU;

    TraceLog(LOG_INFO, "  CUDA available: %s",
        cudaOk ? "YES" : "NO");

    if (cudaOk)
    {
        TraceLog(LOG_INFO, "  CUDA device count: %d",
            (int)torch::cuda::device_count());
        try
        {
            torch::Tensor dummy =
                torch::ones({ 8, 8 }, torch::kCUDA);
            torch::Tensor dummy2 = dummy * 2.0f;
            TraceLog(LOG_INFO,
                "  CUDA kernels tested successfully.");
        }
        catch (const std::exception& e)
        {
            TraceLog(LOG_ERROR,
                "  CUDA kernel execution failed: %s",
                e.what());
            TraceLog(LOG_INFO,
                "  Falling back to CPU training.");
            state->device = torch::kCPU;
        }
    }

    TraceLog(LOG_INFO, "  Selected device: %s",
        state->device.is_cuda() ? "CUDA" : "CPU");
}

//---------------------------------------------------------
// feature autoencoder
//---------------------------------------------------------

static inline void NetworkInitFeatureAE(
    NetworkState* state,
    int inputDim,
    int latentDim)
{
    if (inputDim <= 0) return;

    NetworkEnsureDevice(state);

    // encoder: D -> 128 -> 64 -> L
    // decoder: L -> 64 -> 128 -> D
    state->featuresAutoEncoder = torch::nn::Sequential(
        torch::nn::Linear(inputDim, 128),
        NN_ACTIVATION(),
        torch::nn::Linear(128, 64),
        NN_ACTIVATION(),
        torch::nn::Linear(64, latentDim),
        torch::nn::Linear(latentDim, 64),
        NN_ACTIVATION(),
        torch::nn::Linear(64, 128),
        NN_ACTIVATION(),
        torch::nn::Linear(128, inputDim)
    );

    state->featuresAutoEncoder->to(state->device);

    state->featureAEOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->featuresAutoEncoder->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->featureAELoss = 0.0f;
    state->featureAEIterations = 0;

    TraceLog(LOG_INFO,
        "Feature AE initialized: %d -> %d -> %d",
        inputDim, latentDim, inputDim);
}

static inline void NetworkSaveFeatureAE(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->featuresAutoEncoder) return;

    const std::string path =
        folderPath + "/featureAutoEncoder.bin";
    try
    {
        torch::save(state->featuresAutoEncoder, path);
        TraceLog(LOG_INFO,
            "Saved Feature AE to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Feature AE: %s", e.what());
    }
}

static inline void NetworkLoadFeatureAE(
    NetworkState* state,
    int inputDim,
    int latentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/featureAutoEncoder.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitFeatureAE(state, inputDim, latentDim);
        torch::load(state->featuresAutoEncoder, path);
        state->featuresAutoEncoder->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Feature AE from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Feature AE: %s", e.what());
    }
}


//---------------------------------------------------------
// pose autoencoder
//---------------------------------------------------------

static inline void NetworkInitPoseAE(
    NetworkState* state,
    int inputDim,
    int latentDim)
{
    if (inputDim <= 0) return;

    NetworkEnsureDevice(state);

    // D -> 256 -> 256 -> 16 -> 256 -> 256 -> D
    state->poseAutoEncoder = torch::nn::Sequential(
        torch::nn::Linear(inputDim, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, latentDim),
        torch::nn::Linear(latentDim, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, inputDim)
    );

    state->poseAutoEncoder->to(state->device);

    state->poseAEOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->poseAutoEncoder->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->poseAELoss = 0.0f;
    state->poseAEIterations = 0;

    TraceLog(LOG_INFO,
        "Pose AE initialized: %d -> 256 -> 256 -> %d -> 256 -> 256 -> %d",
        inputDim, latentDim, inputDim);
}

static inline void NetworkSavePoseAE(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->poseAutoEncoder) return;

    const std::string path =
        folderPath + "/poseAutoEncoder.bin";
    try
    {
        torch::save(state->poseAutoEncoder, path);
        TraceLog(LOG_INFO,
            "Saved Pose AE to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Pose AE: %s", e.what());
    }
}
static inline void NetworkLoadPoseAE(
    NetworkState* state,
    int inputDim,
    int latentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/poseAutoEncoder.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitPoseAE(state, inputDim, latentDim);
        torch::load(state->poseAutoEncoder, path);
        state->poseAutoEncoder->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Pose AE from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Pose AE: %s", e.what());
    }
}
//---------------------------------------------------------
// segment autoencoder
//---------------------------------------------------------

static inline void NetworkInitSegmentAE(
    NetworkState* state,
    int flatDim,
    int latentDim)
{
    if (flatDim <= 0) return;

    NetworkEnsureDevice(state);

    // flatDim -> 512 -> 256 -> L -> 256 -> 512 -> flatDim
    state->segmentAutoEncoder = torch::nn::Sequential(
        torch::nn::Linear(flatDim, 512),
        NN_ACTIVATION(),
        torch::nn::Linear(512, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, latentDim),
        torch::nn::Linear(latentDim, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, 512),
        NN_ACTIVATION(),
        torch::nn::Linear(512, flatDim)
    );

    state->segmentAutoEncoder->to(state->device);

    state->segmentOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->segmentAutoEncoder->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->segmentAELoss = 0.0f;
    state->segmentAEIterations = 0;

    TraceLog(LOG_INFO,
        "Segment AE initialized: %d -> 512 -> 256 -> %d"
        " -> 256 -> 512 -> %d",
        flatDim, latentDim, flatDim);
}

static inline void NetworkSaveSegmentAE(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->segmentAutoEncoder) return;

    const std::string path =
        folderPath + "/segmentAutoEncoder.bin";
    try
    {
        torch::save(state->segmentAutoEncoder, path);
        TraceLog(LOG_INFO,
            "Saved Segment AE to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Segment AE: %s", e.what());
    }
}

static inline void NetworkLoadSegmentAE(
    NetworkState* state,
    int flatDim,
    int latentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/segmentAutoEncoder.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitSegmentAE(state, flatDim, latentDim);
        torch::load(state->segmentAutoEncoder, path);
        state->segmentAutoEncoder->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Segment AE from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Segment AE: %s", e.what());
    }
}

//---------------------------------------------------------
// segment latent average predictor
//---------------------------------------------------------

static inline void NetworkInitLatentSegmentPredictor(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim)
{
    NetworkEnsureDevice(state);

    // featureLatent -> 128 -> 256 -> segmentLatent
    state->segmentLatentAveragePredictor =
        torch::nn::Sequential(
            torch::nn::Linear(featureLatentDim, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, segmentLatentDim)
        );

    state->segmentLatentAveragePredictor->to(
        state->device);

    state->latentSegmentPredictorOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->segmentLatentAveragePredictor
                ->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->latentSegmentPredictorLoss = 0.0f;
    state->latentSegmentPredictorIterations = 0;

    TraceLog(LOG_INFO,
        "Latent Predictor initialized: %d -> 128 -> 256"
        " -> %d",
        featureLatentDim, segmentLatentDim);
}

static inline void NetworkSaveLatentSegmentPredictor(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->segmentLatentAveragePredictor) return;

    const std::string path =
        folderPath + "/segmentLatentAveragePredictor.bin";
    try
    {
        torch::save(
            state->segmentLatentAveragePredictor, path);
        TraceLog(LOG_INFO,
            "Saved Latent Predictor to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Latent Predictor: %s",
            e.what());
    }
}

static inline void NetworkLoadLatentSegmentPredictor(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/segmentLatentAveragePredictor.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitLatentSegmentPredictor(
            state, featureLatentDim, segmentLatentDim);
        torch::load(
            state->segmentLatentAveragePredictor, path);
        state->segmentLatentAveragePredictor->to(
            state->device);
        TraceLog(LOG_INFO,
            "Loaded Latent Predictor from: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Latent Predictor: %s",
            e.what());
    }
}

//---------------------------------------------------------
// single pose latent predictor: features -> pose latent
//---------------------------------------------------------

static inline void NetworkInitSinglePosePredictor(
    NetworkState* state,
    int featureLatentDim,
    int poseLatentDim)
{
    NetworkEnsureDevice(state);

    state->singlePosePredictor =
        torch::nn::Sequential(
            torch::nn::Linear(featureLatentDim, 128),
            NN_ACTIVATION(),
            torch::nn::Linear(128, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, poseLatentDim)
        );

    state->singlePosePredictor->to(state->device);

    state->singlePosePredictorOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->singlePosePredictor->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->singlePosePredictorLoss = 0.0f;
    state->singlePosePredictorIterations = 0;

    TraceLog(LOG_INFO,
        "SinglePosePredictor initialized: %d -> 128 -> 256 -> 256 -> %d",
        featureLatentDim, poseLatentDim);
}

static inline void NetworkInitUncondAdvance(
    NetworkState* state,
    int poseLatentDim)
{
    NetworkEnsureDevice(state);

    state->uncondAdvancePredictor =
        torch::nn::Sequential(
            torch::nn::Linear(poseLatentDim, 128),
            NN_ACTIVATION(),
            torch::nn::Linear(128, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, 256),
            NN_ACTIVATION(),
            torch::nn::Linear(256, poseLatentDim)
        );

    state->uncondAdvancePredictor->to(state->device);

    state->uncondAdvanceOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->uncondAdvancePredictor->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->uncondAdvanceLoss = 0.0f;
    state->uncondAdvanceIterations = 0;

    TraceLog(LOG_INFO,
        "UncondAdvance initialized: %d -> 128 -> 256 -> 256 -> %d",
        poseLatentDim, poseLatentDim);
}

static inline void NetworkSaveSinglePosePredictor(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->singlePosePredictor) return;

    const std::string path = folderPath + "/singlePosePredictor.bin";
    try
    {
        torch::save(state->singlePosePredictor, path);
        TraceLog(LOG_INFO, "Saved SinglePosePredictor to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save SinglePosePredictor: %s", e.what());
    }
}

static inline void NetworkLoadSinglePosePredictor(
    NetworkState* state,
    int featureLatentDim,
    int poseLatentDim,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/singlePosePredictor.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitSinglePosePredictor(state, featureLatentDim, poseLatentDim);
        torch::load(state->singlePosePredictor, path);
        state->singlePosePredictor->to(state->device);
        TraceLog(LOG_INFO, "Loaded SinglePosePredictor from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load SinglePosePredictor: %s", e.what());
    }
}

static inline void NetworkSaveUncondAdvance(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->uncondAdvancePredictor) return;

    const std::string path = folderPath + "/uncondAdvance.bin";
    try
    {
        torch::save(state->uncondAdvancePredictor, path);
        TraceLog(LOG_INFO, "Saved UncondAdvance to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save UncondAdvance: %s", e.what());
    }
}

static inline void NetworkLoadUncondAdvance(
    NetworkState* state,
    int poseLatentDim,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/uncondAdvance.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitUncondAdvance(state, poseLatentDim);
        torch::load(state->uncondAdvancePredictor, path);
        state->uncondAdvancePredictor->to(state->device);
        TraceLog(LOG_INFO, "Loaded UncondAdvance from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load UncondAdvance: %s", e.what());
    }
}

//---------------------------------------------------------
// monday predictor: (features, pose latent) → pose latent delta
//---------------------------------------------------------

static inline void NetworkInitMonday(
    NetworkState* state,
    int featureLatentDim,
    int poseLatentDim)
{
    NetworkEnsureDevice(state);

    const int inputDim = featureLatentDim + poseLatentDim;
    state->mondayPredictor = torch::nn::Sequential(
        torch::nn::Linear(inputDim, 128),
        NN_ACTIVATION(),
        torch::nn::Linear(128, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, 256),
        NN_ACTIVATION(),
        torch::nn::Linear(256, poseLatentDim)
    );

    state->mondayPredictor->to(state->device);

    state->mondayOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->mondayPredictor->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->mondayLoss = 0.0f;
    state->mondayIterations = 0;

    TraceLog(LOG_INFO,
        "Monday initialized: (%d+%d) -> 128 -> 256 -> 256 -> %d",
        featureLatentDim, poseLatentDim, poseLatentDim);
}

constexpr double MONDAY_DELTA_STATS_INTERVAL = 20.0;

// sample consecutive frame pairs, encode through PoseAE, compute per-dim mean/std of deltas
static inline void ComputeMondayDeltaStats(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (!state->poseAutoEncoder) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int sampleCount = std::min(2000, (int)db->legalStartFrames.size());
    const int poseDim = db->poseGenFeaturesComputeDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->poseAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // encode consecutive frame pairs
    torch::Tensor poseBatch0 = torch::empty({sampleCount, poseDim});
    torch::Tensor poseBatch1 = torch::empty({sampleCount, poseDim});
    float* p0Ptr = poseBatch0.data_ptr<float>();
    float* p1Ptr = poseBatch1.data_ptr<float>();

    for (int i = 0; i < sampleCount; ++i)
    {
        const int frame = SampleLegalFrame(db);
        std::span<const float> row0 = db->normalizedPoseGenFeatures.row_view(frame);
        std::copy(row0.begin(), row0.end(), p0Ptr + i * poseDim);
        std::span<const float> row1 = db->normalizedPoseGenFeatures.row_view(frame + 1);
        std::copy(row1.begin(), row1.end(), p1Ptr + i * poseDim);
    }

    poseBatch0 = poseBatch0.to(state->device);
    poseBatch1 = poseBatch1.to(state->device);

    torch::Tensor z0 = PoseAEEncode(state->poseAutoEncoder, poseBatch0);
    torch::Tensor z1 = PoseAEEncode(state->poseAutoEncoder, poseBatch1);
    torch::Tensor deltas = (z1 - z0).to(torch::kCPU);

    torch::Tensor meanTensor = deltas.mean(0);
    torch::Tensor stdTensor = deltas.std(0);

    // clamp std to avoid division by zero
    stdTensor = torch::clamp(stdTensor, 1e-6f);

    state->mondayDeltaMean.resize(latentDim);
    state->mondayDeltaStd.resize(latentDim);
    const float* mPtr = meanTensor.data_ptr<float>();
    const float* sPtr = stdTensor.data_ptr<float>();
    for (int d = 0; d < latentDim; ++d)
    {
        state->mondayDeltaMean[d] = mPtr[d];
        state->mondayDeltaStd[d] = sPtr[d];
    }

    TraceLog(LOG_INFO,
        "Monday delta stats: %d samples, mean_std=%.6f, mean_mean=%.6f",
        sampleCount, stdTensor.mean().item<float>(), meanTensor.mean().item<float>());
}

static inline void SaveMondayDeltaStats(
    const NetworkState* state,
    const std::string& folder)
{
    if (state->mondayDeltaMean.empty()) return;
    const std::string path = folder + "/mondayDeltaStats.bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    const int n = (int)state->mondayDeltaMean.size();
    fwrite(&n, sizeof(int), 1, f);
    fwrite(state->mondayDeltaMean.data(), sizeof(float), n, f);
    fwrite(state->mondayDeltaStd.data(), sizeof(float), n, f);
    fclose(f);
}

static inline void LoadMondayDeltaStats(
    NetworkState* state,
    const std::string& folder)
{
    const std::string path = folder + "/mondayDeltaStats.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0 || n > 1024)
    {
        fclose(f);
        return;
    }
    state->mondayDeltaMean.resize(n);
    state->mondayDeltaStd.resize(n);
    if (fread(state->mondayDeltaMean.data(), sizeof(float), n, f) != (size_t)n
        || fread(state->mondayDeltaStd.data(), sizeof(float), n, f) != (size_t)n)
    {
        state->mondayDeltaMean.clear();
        state->mondayDeltaStd.clear();
    }
    fclose(f);
    if (!state->mondayDeltaMean.empty())
    {
        TraceLog(LOG_INFO, "Loaded Monday delta stats (%d dims)", n);
    }
}

static inline void NetworkSaveMonday(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->mondayPredictor) return;

    const std::string path = folderPath + "/monday.bin";
    try
    {
        torch::save(state->mondayPredictor, path);
        SaveMondayDeltaStats(state, folderPath);
        TraceLog(LOG_INFO, "Saved Monday to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save Monday: %s", e.what());
    }
}

static inline void NetworkLoadMonday(
    NetworkState* state,
    int featureLatentDim,
    int poseLatentDim,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/monday.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitMonday(state, featureLatentDim, poseLatentDim);
        torch::load(state->mondayPredictor, path);
        state->mondayPredictor->to(state->device);
        LoadMondayDeltaStats(state, folderPath);
        TraceLog(LOG_INFO, "Loaded Monday from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load Monday: %s", e.what());
    }
}

//---------------------------------------------------------
// latent space statistics (for debugging/analysis)
//---------------------------------------------------------

static inline void NetworkAnalyzeLatentSpaceStatistics(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    TraceLog(LOG_INFO, "=== Analyzing Latent Space Statistics ===");

    const int sampleCount = std::min(1000, (int)db->legalStartFrames.size());
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;

    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();

    // accumulate latent values for statistics
    std::vector<float> featureLatentValues;
    std::vector<float> segmentLatentValues;
    featureLatentValues.reserve(sampleCount * FEATURE_AE_LATENT_DIM);
    segmentLatentValues.reserve(sampleCount * SEGMENT_AE_LATENT_DIM);

    try
    {
        for (int i = 0; i < sampleCount; ++i)
        {
            const int ri = RandomInt((int)db->legalStartFrames.size());
            const int frame = db->legalStartFrames[ri];

            // build feature batch
            torch::Tensor featureTensor = torch::empty({ 1, featureDim });
            float* fPtr = featureTensor.data_ptr<float>();
            std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
            std::copy(fRow.begin(), fRow.end(), fPtr);

            // build segment batch
            torch::Tensor segmentTensor = torch::empty({ 1, flatDim });
            float* sPtr = segmentTensor.data_ptr<float>();
            const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
            memcpy(sPtr, src, (size_t)segFrames * pgDim * sizeof(float));

            featureTensor = featureTensor.to(state->device);
            segmentTensor = segmentTensor.to(state->device);

            torch::Tensor featureLatent;
            torch::Tensor segmentLatent;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureTensor);
                segmentLatent = EncodeWithAE(state->segmentAutoEncoder, segmentTensor);
            }

            // copy to CPU and accumulate
            featureLatent = featureLatent.to(torch::kCPU);
            segmentLatent = segmentLatent.to(torch::kCPU);

            const float* fLatentPtr = featureLatent.data_ptr<float>();
            const float* sLatentPtr = segmentLatent.data_ptr<float>();

            for (int d = 0; d < FEATURE_AE_LATENT_DIM; ++d)
            {
                featureLatentValues.push_back(fLatentPtr[d]);
            }

            for (int d = 0; d < SEGMENT_AE_LATENT_DIM; ++d)
            {
                segmentLatentValues.push_back(sLatentPtr[d]);
            }
        }

        // compute mean
        float featureMean = 0.0f;
        for (float v : featureLatentValues) featureMean += v;
        featureMean /= (float)featureLatentValues.size();

        float segmentMean = 0.0f;
        for (float v : segmentLatentValues) segmentMean += v;
        segmentMean /= (float)segmentLatentValues.size();

        // compute variance and std
        float featureVariance = 0.0f;
        for (float v : featureLatentValues)
        {
            const float diff = v - featureMean;
            featureVariance += diff * diff;
        }
        featureVariance /= (float)featureLatentValues.size();
        const float featureStd = std::sqrt(featureVariance);

        float segmentVariance = 0.0f;
        for (float v : segmentLatentValues)
        {
            const float diff = v - segmentMean;
            segmentVariance += diff * diff;
        }
        segmentVariance /= (float)segmentLatentValues.size();
        const float segmentStd = std::sqrt(segmentVariance);

        // compute min/max
        const float featureMin = *std::min_element(
            featureLatentValues.begin(), featureLatentValues.end());
        const float featureMax = *std::max_element(
            featureLatentValues.begin(), featureLatentValues.end());

        const float segmentMin = *std::min_element(
            segmentLatentValues.begin(), segmentLatentValues.end());
        const float segmentMax = *std::max_element(
            segmentLatentValues.begin(), segmentLatentValues.end());

        TraceLog(LOG_INFO, "Feature AE Latent Space (dim=%d, samples=%d):",
            FEATURE_AE_LATENT_DIM, sampleCount);
        TraceLog(LOG_INFO, "  Mean: %.6f", featureMean);
        TraceLog(LOG_INFO, "  Std:  %.6f", featureStd);
        TraceLog(LOG_INFO, "  Min:  %.6f", featureMin);
        TraceLog(LOG_INFO, "  Max:  %.6f", featureMax);

        TraceLog(LOG_INFO, "Segment AE Latent Space (dim=%d, samples=%d):",
            SEGMENT_AE_LATENT_DIM, sampleCount);
        TraceLog(LOG_INFO, "  Mean: %.6f", segmentMean);
        TraceLog(LOG_INFO, "  Std:  %.6f", segmentStd);
        TraceLog(LOG_INFO, "  Min:  %.6f", segmentMin);
        TraceLog(LOG_INFO, "  Max:  %.6f", segmentMax);

        TraceLog(LOG_INFO, "=== Analysis Complete ===");
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Latent space analysis error: %s", e.what());
    }
}


//---------------------------------------------------------
// FlowModel implementation
//---------------------------------------------------------

inline FlowModelImpl::FlowModelImpl(int inputDim, int hiddenDim, int outputDim)
    : inputProj(register_module("inputProj", torch::nn::Linear(inputDim, hiddenDim))),
    res1a(register_module("res1a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res1b(register_module("res1b", torch::nn::Linear(hiddenDim, hiddenDim))),
    res2a(register_module("res2a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res2b(register_module("res2b", torch::nn::Linear(hiddenDim, hiddenDim))),
    res3a(register_module("res3a", torch::nn::Linear(hiddenDim, hiddenDim))),
    res3b(register_module("res3b", torch::nn::Linear(hiddenDim, hiddenDim))),
    outputProj(register_module("outputProj", torch::nn::Linear(hiddenDim, outputDim)))
{
}

inline torch::Tensor FlowModelImpl::forward(const torch::Tensor& x)
{
    // Input projection: inputDim -> hiddenDim
    torch::Tensor h = nn_activate(inputProj->forward(x));

    // Residual block 1: x = x + gelu(res1b(gelu(res1a(x))))
    h = h + nn_activate(res1b->forward(nn_activate(res1a->forward(h))));

    // Residual block 2
    h = h + nn_activate(res2b->forward(nn_activate(res2a->forward(h))));

    // Residual block 3
    h = h + nn_activate(res3b->forward(nn_activate(res3a->forward(h))));

    // Output projection: hiddenDim -> outputDim
    return outputProj->forward(h);
}

//---------------------------------------------------------
// full-space flow model (condition re-injected at each layer)
//---------------------------------------------------------

inline FullFlowModelImpl::FullFlowModelImpl(int featureDim, int flatDim)
    : layer1(register_module("layer1",
          torch::nn::Linear(flatDim + featureDim + 1, 256))),
      layer2(register_module("layer2",
          torch::nn::Linear(256 + featureDim + 1, 512))),
      outputLayer(register_module("outputLayer",
          torch::nn::Linear(512 + featureDim + 1, flatDim))),
      condTimeDim(featureDim + 1)
{
}

inline torch::Tensor FullFlowModelImpl::forward(
    const torch::Tensor& xt,
    const torch::Tensor& condTime)
{
    torch::Tensor h = nn_activate(
        layer1->forward(torch::cat({xt, condTime}, 1)));
    h = nn_activate(
        layer2->forward(torch::cat({h, condTime}, 1)));
    return outputLayer->forward(torch::cat({h, condTime}, 1));
}

//---------------------------------------------------------
// friday flow: frame-by-frame flow in pose latent space
//---------------------------------------------------------

inline FridayFlowModelImpl::FridayFlowModelImpl(int featureDim, int poseLatentDim)
    : layer1(register_module("layer1",
          torch::nn::Linear(poseLatentDim + featureDim + poseLatentDim + 1, 128))),
      layer2(register_module("layer2",
          torch::nn::Linear(128 + featureDim + poseLatentDim + 1, 256))),
      outputLayer(register_module("outputLayer",
          torch::nn::Linear(256 + featureDim + poseLatentDim + 1, poseLatentDim))),
      condTimeDim(featureDim + poseLatentDim + 1)
{
}

inline torch::Tensor FridayFlowModelImpl::forward(
    const torch::Tensor& xt, const torch::Tensor& condTime)
{
    torch::Tensor h = nn_activate(layer1->forward(torch::cat({xt, condTime}, 1)));
    h = nn_activate(layer2->forward(torch::cat({h, condTime}, 1)));
    return outputLayer->forward(torch::cat({h, condTime}, 1));
}

// compute per-dimension std of pose latent codes by sampling from the database
// call after PoseAE is trained (headstart done)
static inline void ComputePoseLatentStats(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (!state->poseAutoEncoder) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int sampleCount = std::min(2000, (int)db->legalStartFrames.size());
    const int poseDim = db->poseGenFeaturesComputeDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->poseAutoEncoder->eval();

    // encode a bunch of poses and collect latent values
    torch::Tensor allLatents = torch::empty({sampleCount, latentDim});

    {
        torch::NoGradGuard noGrad;
        torch::Tensor poseBatch = torch::empty({sampleCount, poseDim});
        float* pPtr = poseBatch.data_ptr<float>();

        for (int i = 0; i < sampleCount; ++i)
        {
            const int frame = db->legalStartFrames[
                RandomInt((int)db->legalStartFrames.size())];
            std::span<const float> row =
                db->normalizedPoseGenFeatures.row_view(frame);
            std::copy(row.begin(), row.end(), pPtr + i * poseDim);
        }

        poseBatch = poseBatch.to(state->device);
        allLatents = PoseAEEncode(state->poseAutoEncoder, poseBatch);
        allLatents = allLatents.to(torch::kCPU);
    }

    // compute per-dim std
    torch::Tensor stdTensor = allLatents.std(0);
    state->poseLatentStd.resize(latentDim);
    const float* sPtr = stdTensor.data_ptr<float>();
    for (int d = 0; d < latentDim; ++d)
    {
        state->poseLatentStd[d] = sPtr[d];
    }

    TraceLog(LOG_INFO,
        "Pose latent stats computed from %d samples (mean std=%.4f)",
        sampleCount, stdTensor.mean().item<float>());
}

static inline void SavePoseLatentStats(
    const NetworkState* state,
    const std::string& folder)
{
    if (state->poseLatentStd.empty()) return;
    const std::string path = folder + "/poseLatentStd.bin";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    const int n = (int)state->poseLatentStd.size();
    fwrite(&n, sizeof(int), 1, f);
    fwrite(state->poseLatentStd.data(), sizeof(float), n, f);
    fclose(f);
}

static inline void LoadPoseLatentStats(
    NetworkState* state,
    const std::string& folder)
{
    const std::string path = folder + "/poseLatentStd.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0 || n > 1024)
    {
        fclose(f);
        return;
    }
    state->poseLatentStd.resize(n);
    if (fread(state->poseLatentStd.data(), sizeof(float), n, f) != (size_t)n)
    {
        state->poseLatentStd.clear();
    }
    fclose(f);
    if (!state->poseLatentStd.empty())
    {
        TraceLog(LOG_INFO, "Loaded pose latent stats (%d dims) from: %s",
            n, path.c_str());
    }
}

static inline void NetworkInitFridayFlow(NetworkState* state, int featureDim, int poseLatentDim)
{
    if (featureDim <= 0 || poseLatentDim <= 0) return;
    NetworkEnsureDevice(state);

    state->fridayFlowModel = FridayFlowModel(featureDim, poseLatentDim);
    state->fridayFlowModel->to(state->device);

    state->fridayFlowOptimizer = std::make_shared<torch::optim::Adam>(
        state->fridayFlowModel->parameters(), torch::optim::AdamOptions(1e-3));

    state->fridayFlowLoss = 0.0f;
    state->fridayFlowIterations = 0;
    TraceLog(LOG_INFO, "FridayFlow initialized: featureDim=%d poseLatentDim=%d",
        featureDim, poseLatentDim);
}

static inline void NetworkSaveFridayFlow(const NetworkState* state, const std::string& folder)
{
    if (!state->fridayFlowModel) return;
    const std::string path = folder + "/fridayFlowModel.bin";
    torch::save(state->fridayFlowModel, path);
    SavePoseLatentStats(state, folder);
    TraceLog(LOG_INFO, "Saved FridayFlow to: %s", path.c_str());
}

static inline void NetworkLoadFridayFlow(
    NetworkState* state,
    const std::string& folder,
    int featureDim,
    int poseLatentDim,
    const AnimDatabase* db)
{
    const std::string path = folder + "/fridayFlowModel.bin";
    try
    {
        NetworkInitFridayFlow(state, featureDim, poseLatentDim);
        torch::load(state->fridayFlowModel, path);
        state->fridayFlowModel->to(state->device);
        TraceLog(LOG_INFO, "Loaded FridayFlow from: %s", path.c_str());

        // load or recompute pose latent stats
        LoadPoseLatentStats(state, folder);
        if (state->poseLatentStd.empty() && db != nullptr)
        {
            ComputePoseLatentStats(state, db);
            SavePoseLatentStats(state, folder);
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_WARNING, "Could not load FridayFlow: %s", e.what());
        state->fridayFlowModel = nullptr;
        state->fridayFlowOptimizer = nullptr;
    }
}

//---------------------------------------------------------
// latent flow matching model
//---------------------------------------------------------

static inline void NetworkInitFlow(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim)
{
    NetworkEnsureDevice(state);

    // input: condition + xt + time
    // condition = featureLatent + avgLatent (if DO_DELTA_FLOWMATCHING) or just featureLatent
#ifdef DO_DELTA_FLOWMATCHING
    const int inputDim = featureLatentDim + segmentLatentDim + segmentLatentDim + 1;
#else
    const int inputDim = featureLatentDim + segmentLatentDim + 1;
#endif

    state->latentFlowModel = FlowModel(inputDim, 256, segmentLatentDim);
    //state->latentFlowModel = torch::nn::Sequential(
    //    torch::nn::Linear(inputDim, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, 256),
    //    NN_ACTIVATION(),
    //    torch::nn::Linear(256, segmentLatentDim));

    state->latentFlowModel->to(state->device);

    state->flowOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->latentFlowModel->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->flowLoss = 0.0f;
    state->flowIterations = 0;

    TraceLog(LOG_INFO,
        "Latent Flow Model initialized: %d -> 256 (3 residual blocks) -> %d",
        inputDim, segmentLatentDim);
}

static inline void NetworkSaveFlow(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->latentFlowModel) return;

    const std::string path =
        folderPath + "/latentFlowModel.bin";
    try
    {
        torch::save(state->latentFlowModel, path);
        TraceLog(LOG_INFO,
            "Saved Latent Flow Model to: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to save Latent Flow Model: %s",
            e.what());
    }
}

static inline void NetworkLoadFlow(
    NetworkState* state,
    int featureLatentDim,
    int segmentLatentDim,
    const std::string& folderPath)
{
    const std::string path =
        folderPath + "/latentFlowModel.bin";
    if (!std::filesystem::exists(path)) return;

    try
    {
        NetworkInitFlow(
            state, featureLatentDim, segmentLatentDim);
        torch::load(state->latentFlowModel, path);
        state->latentFlowModel->to(state->device);
        TraceLog(LOG_INFO,
            "Loaded Latent Flow Model from: %s",
            path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Failed to load Latent Flow Model: %s",
            e.what());
    }
}

//---------------------------------------------------------
// full-space flow model init / save / load
//---------------------------------------------------------

static inline void NetworkInitFullFlow(
    NetworkState* state,
    int featureDim,
    int flatDim)
{
    if (featureDim <= 0 || flatDim <= 0) return;

    NetworkEnsureDevice(state);

    state->fullFlowModel = FullFlowModel(featureDim, flatDim);
    state->fullFlowModel->to(state->device);

    state->fullFlowOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->fullFlowModel->parameters(),
            torch::optim::AdamOptions(1e-3));

    state->fullFlowLoss = 0.0f;
    state->fullFlowIterations = 0;

    TraceLog(LOG_INFO,
        "FullFlow initialized: cond=%d flat=%d "
        "layers: %d->256->512->%d (cond re-injected)",
        featureDim, flatDim,
        flatDim + featureDim + 1, flatDim);
}

static inline void NetworkSaveFullFlow(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->fullFlowModel) return;
    const std::string path = folderPath + "/fullFlowModel.bin";
    try
    {
        torch::save(state->fullFlowModel, path);
        TraceLog(LOG_INFO, "Saved FullFlow to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save FullFlow: %s", e.what());
    }
}

static inline void NetworkLoadFullFlow(
    NetworkState* state,
    int featureDim,
    int flatDim,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/fullFlowModel.bin";
    if (!std::filesystem::exists(path)) return;
    try
    {
        NetworkInitFullFlow(state, featureDim, flatDim);
        torch::load(state->fullFlowModel, path);
        state->fullFlowModel->to(state->device);
        TraceLog(LOG_INFO, "Loaded FullFlow from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load FullFlow: %s", e.what());
    }
}

//---------------------------------------------------------
// loss history CSV save/load
//---------------------------------------------------------

static inline void LossHistorySave(
    const NetworkState* state,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/loss_history.csv";
    FILE* f = fopen(path.c_str(), "w");
    if (!f)
    {
        TraceLog(LOG_ERROR, "Failed to write loss history: %s", path.c_str());
        return;
    }

    fprintf(f, "time,featureAE,poseAE,segmentAE,predictor,flow,segmentE2e,fullFlow,fridayFlow,singlePose,singlePoseE2e,uncondAdvance,monday\n");
    const int n = (int)state->lossHistoryTime.size();
    for (int i = 0; i < n; ++i)
    {
        fprintf(f, "%.1f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            state->lossHistoryTime[i],
            i < (int)state->featureAELossHistory.size() ? state->featureAELossHistory[i] : 0.0f,
            i < (int)state->poseAELossHistory.size() ? state->poseAELossHistory[i] : 0.0f,
            i < (int)state->segmentAELossHistory.size() ? state->segmentAELossHistory[i] : 0.0f,
            i < (int)state->latentSegmentPredictorLossHistory.size() ? state->latentSegmentPredictorLossHistory[i] : 0.0f,
            i < (int)state->flowLossHistory.size() ? state->flowLossHistory[i] : 0.0f,
            i < (int)state->segmentE2eLossHistory.size() ? state->segmentE2eLossHistory[i] : 0.0f,
            i < (int)state->fullFlowLossHistory.size() ? state->fullFlowLossHistory[i] : 0.0f,
            i < (int)state->fridayFlowLossHistory.size() ? state->fridayFlowLossHistory[i] : 0.0f,
            i < (int)state->singlePosePredictorLossHistory.size() ? state->singlePosePredictorLossHistory[i] : 0.0f,
            i < (int)state->singlePoseE2eLossHistory.size() ? state->singlePoseE2eLossHistory[i] : 0.0f,
            i < (int)state->uncondAdvanceLossHistory.size() ? state->uncondAdvanceLossHistory[i] : 0.0f,
            i < (int)state->mondayLossHistory.size() ? state->mondayLossHistory[i] : 0.0f);
    }

    fclose(f);
    TraceLog(LOG_INFO, "Saved loss history (%d points) to: %s", n, path.c_str());
}

static inline void LossHistoryLoad(
    NetworkState* state,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/loss_history.csv";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return;

    state->lossHistoryTime.clear();
    state->featureAELossHistory.clear();
    state->poseAELossHistory.clear();
    state->segmentAELossHistory.clear();
    state->latentSegmentPredictorLossHistory.clear();
    state->singlePosePredictorLossHistory.clear();
    state->flowLossHistory.clear();
    state->segmentE2eLossHistory.clear();
    state->fullFlowLossHistory.clear();
    state->fridayFlowLossHistory.clear();
    state->singlePoseE2eLossHistory.clear();
    state->uncondAdvanceLossHistory.clear();
    state->mondayLossHistory.clear();

    // skip header
    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return; }

    // columns: time,featureAE,poseAE,segmentAE,predictor,flow,segmentE2e,fullFlow,fridayFlow,singlePose,singlePoseE2e,uncondAdvance,monday
    while (fgets(line, sizeof(line), f))
    {
        float t = 0, fae = 0, pae = 0, sae = 0;
        float pred = 0, flow = 0, segE2e = 0;
        float ff = 0, friday = 0, singlePose = 0, singlePoseE2e = 0;
        float uncondAdv = 0, monday = 0;
        const int cols = sscanf(
            line,
            "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
            &t, &fae, &pae, &sae,
            &pred, &flow, &segE2e, &ff, &friday,
            &singlePose, &singlePoseE2e, &uncondAdv, &monday);
        if (cols < 5) break;
        state->lossHistoryTime.push_back(t);
        state->featureAELossHistory.push_back(fae);
        state->poseAELossHistory.push_back(cols >= 3 ? pae : 0.0f);
        state->segmentAELossHistory.push_back(cols >= 4 ? sae : 0.0f);
        state->latentSegmentPredictorLossHistory.push_back(cols >= 5 ? pred : 0.0f);
        state->flowLossHistory.push_back(cols >= 6 ? flow : 0.0f);
        state->segmentE2eLossHistory.push_back(cols >= 7 ? segE2e : 0.0f);
        state->fullFlowLossHistory.push_back(cols >= 8 ? ff : 0.0f);
        state->fridayFlowLossHistory.push_back(cols >= 9 ? friday : 0.0f);
        state->singlePosePredictorLossHistory.push_back(cols >= 10 ? singlePose : 0.0f);
        state->singlePoseE2eLossHistory.push_back(cols >= 11 ? singlePoseE2e : 0.0f);
        state->uncondAdvanceLossHistory.push_back(cols >= 12 ? uncondAdv : 0.0f);
        state->mondayLossHistory.push_back(cols >= 13 ? monday : 0.0f);
    }

    fclose(f);
    if (!state->lossHistoryTime.empty())
    {
        TraceLog(LOG_INFO, "Loaded loss history (%d points) from: %s",
            (int)state->lossHistoryTime.size(), path.c_str());
    }
}

//---------------------------------------------------------
// save/load all networks at once
//---------------------------------------------------------

static inline void NetworkSaveAll(
    const NetworkState* state,
    const std::string& folderPath)
{
    NetworkSaveFeatureAE(state, folderPath);
    NetworkSavePoseAE(state, folderPath);
    NetworkSaveSegmentAE(state, folderPath);
    NetworkSaveLatentSegmentPredictor(state, folderPath);
    NetworkSaveSinglePosePredictor(state, folderPath);
    NetworkSaveFlow(state, folderPath);
    NetworkSaveFullFlow(state, folderPath);
    NetworkSaveFridayFlow(state, folderPath);
    NetworkSaveUncondAdvance(state, folderPath);
    NetworkSaveMonday(state, folderPath);
    LossHistorySave(state, folderPath);
}

static inline void NetworkLoadAll(
    NetworkState* state,
    const AnimDatabase* db,  
    const std::string& folderPath)
{
    const int featureDim = db->featureDim;
    const int segmentFlatDim = db->poseGenSegmentFlatDim;
    NetworkLoadFeatureAE(state, featureDim, FEATURE_AE_LATENT_DIM, folderPath);
    NetworkLoadPoseAE(state, db->poseGenFeaturesComputeDim, POSE_AE_LATENT_DIM, folderPath); 
    if (segmentFlatDim > 0)
    {
        NetworkLoadSegmentAE(state, segmentFlatDim, SEGMENT_AE_LATENT_DIM, folderPath);
    }
    NetworkLoadLatentSegmentPredictor(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM, folderPath);
    NetworkLoadSinglePosePredictor(state, FEATURE_AE_LATENT_DIM, POSE_AE_LATENT_DIM, folderPath);
    NetworkLoadFlow(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM, folderPath);
    if (segmentFlatDim > 0)
    {
        NetworkLoadFullFlow(state, featureDim, segmentFlatDim, folderPath);
    }
    NetworkLoadFridayFlow(
        state, folderPath, FEATURE_AE_LATENT_DIM,
        POSE_AE_LATENT_DIM, db);
    NetworkLoadUncondAdvance(state, POSE_AE_LATENT_DIM, folderPath);
    NetworkLoadMonday(state, FEATURE_AE_LATENT_DIM, POSE_AE_LATENT_DIM, folderPath);
    LossHistoryLoad(state, folderPath);
}


//---------------------------------------------------------
// time-budgeted training functions
//---------------------------------------------------------


// train feature AE for up to budgetSeconds
static inline void NetworkTrainFeatureAEForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->featureAEOptimizer) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->featureDim <= 0) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const Clock::time_point start = Clock::now();

    state->featuresAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor targetHost =
                torch::empty({ batchSize, featureDim });
            float* ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int idx = SampleLegalFrame(db);
                std::span<const float> row =
                    db->normalizedFeatures.row_view(idx);
                std::copy(
                    row.begin(), row.end(),
                    ptr + b * featureDim);
            }

            torch::Tensor target =
                targetHost.to(state->device);

            // denoising: add gaussian noise
            torch::Tensor noise =
                torch::randn({ batchSize, featureDim })
                    .to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            state->featureAEOptimizer->zero_grad();
            torch::Tensor output =
                state->featuresAutoEncoder->forward(input);
            torch::Tensor loss =
                torch::mse_loss(output, target);

            loss.backward();
            state->featureAEOptimizer->step();

            state->featureAELoss = loss.item<float>();
            state->featureAELossSmoothed = state->featureAELossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->featureAELoss * LOSS_SMOOTHING_FACTOR;
            state->featureAEIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Feature AE Training Error: %s", e.what());
        state->isTraining = false;
    }
}

static inline void NetworkTrainPoseAEForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->poseAutoEncoder) return;
    if (!state->poseAEOptimizer) return;
    if (db->normalizedPoseGenFeatures.empty()) return;
    if (db->poseGenFeaturesComputeDim <= 0) return;

    const int batchSize = 64;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const Clock::time_point start = Clock::now();

    state->poseAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // sample triplets of consecutive frames (prev, curr, next) from same clip
            // so we can encourage straight latent trajectories
            torch::Tensor prevHost = torch::empty({batchSize, poseDim});
            torch::Tensor currHost = torch::empty({batchSize, poseDim});
            torch::Tensor nextHost = torch::empty({batchSize, poseDim});
            float* prevPtr = prevHost.data_ptr<float>();
            float* currPtr = currHost.data_ptr<float>();
            float* nextPtr = nextHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                // legal frames have enough buffer from clip boundaries
                // that frame-1 and frame+1 are always in the same clip
                const int frame = SampleLegalFrame(db);

                std::span<const float> rowPrev =
                    db->normalizedPoseGenFeatures.row_view(frame - 1);
                std::span<const float> rowCurr =
                    db->normalizedPoseGenFeatures.row_view(frame);
                std::span<const float> rowNext =
                    db->normalizedPoseGenFeatures.row_view(frame + 1);
                std::copy(rowPrev.begin(), rowPrev.end(), prevPtr + b * poseDim);
                std::copy(rowCurr.begin(), rowCurr.end(), currPtr + b * poseDim);
                std::copy(rowNext.begin(), rowNext.end(), nextPtr + b * poseDim);
            }

            torch::Tensor prev = prevHost.to(state->device);
            torch::Tensor curr = currHost.to(state->device);
            torch::Tensor next = nextHost.to(state->device);

            // reconstruction loss on curr (with denoising noise, same as before)
            torch::Tensor noise = torch::randn({batchSize, poseDim})
                .to(state->device) * 0.05f;
            torch::Tensor output = state->poseAutoEncoder->forward(curr + noise);
            torch::Tensor reconLoss = torch::mse_loss(output, curr);

            // straightness loss: penalize latent second derivative
            // z_curr should be the midpoint of z_prev and z_next
            torch::Tensor zPrev = PoseAEEncode(state->poseAutoEncoder, prev);
            torch::Tensor zCurr = PoseAEEncode(state->poseAutoEncoder, curr);
            torch::Tensor zNext = PoseAEEncode(state->poseAutoEncoder, next);
            torch::Tensor straightLoss = torch::mean(
                torch::square(zPrev - 2.0f * zCurr + zNext));

            // tightness loss: penalize distances between consecutive latent codes
            // encourages the encoder to place consecutive frames close together
            torch::Tensor tightLoss = torch::mean(torch::square(zCurr - zPrev))
                + torch::mean(torch::square(zNext - zCurr));

            torch::Tensor loss = reconLoss
                + POSE_AE_STRAIGHT_WEIGHT * straightLoss
                + POSE_AE_TIGHT_WEIGHT * tightLoss;

            state->poseAEOptimizer->zero_grad();
            loss.backward();
            state->poseAEOptimizer->step();

            state->poseAELoss = loss.item<float>();
            state->poseAELossSmoothed = state->poseAELossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->poseAELoss * LOSS_SMOOTHING_FACTOR;
            state->poseAEIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Pose AE Training Error: %s", e.what());
        state->isTraining = false;
    }
}

// train segment AE for up to budgetSeconds
static inline void NetworkTrainSegmentAEForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentOptimizer) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int flatDim = db->poseGenSegmentFlatDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const Clock::time_point start = Clock::now();

    state->segmentAutoEncoder->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor targetHost = torch::empty({ batchSize, flatDim });
            float* ptr = targetHost.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int globalStart = SampleLegalFrame(db);

                assert(
                    (globalStart + segFrames)
                    <= db->clipEndFrame[
                        FindClipForMotionFrame(
                            db, globalStart)]);

                float* dst = ptr + b * flatDim;
                const float* src = db->normalizedPoseGenFeatures.data() + globalStart * pgDim;
                memcpy(dst, src,
                    (size_t)segFrames * pgDim * sizeof(float));
            }

            torch::Tensor target = targetHost.to(state->device);

            torch::Tensor noise =
                torch::randn({ batchSize, flatDim })
                    .to(state->device) * 0.05f;
            torch::Tensor input = target + noise;

            state->segmentOptimizer->zero_grad();
            torch::Tensor output =state->segmentAutoEncoder->forward(input);
            torch::Tensor loss = torch::mse_loss(output, target);

            loss.backward();
            state->segmentOptimizer->step();

            state->segmentAELoss = loss.item<float>();
            state->segmentAELossSmoothed = state->segmentAELossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->segmentAELoss * LOSS_SMOOTHING_FACTOR;
            state->segmentAEIterations++;

            //if (state->segmentAEIterations % 1000 == 0)
            //{
            //    // analyze latent space statistics after a bit of AE training
            //    NetworkAnalyzeLatentSpaceStatistics(state, db);
            //}
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Segment AE Training Error: %s", e.what());
        state->isTraining = false;
    }
}

// train predictor for up to budgetSeconds
// both AEs are frozen (eval + NoGrad)
static inline void NetworkTrainLatentSegmentPredictorForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->segmentLatentAveragePredictor) return;
    if (!state->latentSegmentPredictorOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const Clock::time_point start = Clock::now();

    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();
    state->segmentLatentAveragePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // build both batches on CPU
            torch::Tensor featureBatch =
                torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch =
                torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                // feature for this frame
                std::span<const float> fRow =
                    db->normalizedFeatures.row_view(frame);
                std::copy(
                    fRow.begin(), fRow.end(),
                    fPtr + b * featureDim);

                // poseGen segment starting at this frame
                const float* src =
                    db->normalizedPoseGenFeatures.data()
                    + frame * pgDim;
                memcpy(
                    sPtr + b * flatDim, src,
                    (size_t)segFrames * pgDim
                    * sizeof(float));
            }

            featureBatch =
                featureBatch.to(state->device);
            segmentBatch =
                segmentBatch.to(state->device);

            // encode with both AEs (frozen)
            torch::Tensor featureLatent;
            torch::Tensor segmentLatent;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(
                    state->featuresAutoEncoder,
                    featureBatch);
                segmentLatent = EncodeWithAE(
                    state->segmentAutoEncoder,
                    segmentBatch);
            }

            // predict and optimize
            state->latentSegmentPredictorOptimizer->zero_grad();
            torch::Tensor predicted =
                state->segmentLatentAveragePredictor
                    ->forward(featureLatent);
            torch::Tensor loss =
                torch::mse_loss(predicted, segmentLatent);

            loss.backward();
            state->latentSegmentPredictorOptimizer->step();

            state->latentSegmentPredictorLoss = loss.item<float>();
            state->latentSegmentPredictorLossSmoothed = state->latentSegmentPredictorLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->latentSegmentPredictorLoss * LOSS_SMOOTHING_FACTOR;
            state->latentSegmentPredictorIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Predictor Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// single pose predictor training
// featureAE and poseAE are frozen
//---------------------------------------------------------

static inline void NetworkTrainSinglePosePredictorForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->singlePosePredictor) return;
    if (!state->singlePosePredictorOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->poseAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const Clock::time_point start = Clock::now();

    state->featuresAutoEncoder->eval();
    state->poseAutoEncoder->eval();
    state->singlePosePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor featureBatch = torch::empty({batchSize, featureDim});
            torch::Tensor poseBatch = torch::empty({batchSize, poseDim});
            float* fPtr = featureBatch.data_ptr<float>();
            float* pPtr = poseBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                std::span<const float> pRow = db->normalizedPoseGenFeatures.row_view(frame);
                std::copy(pRow.begin(), pRow.end(), pPtr + b * poseDim);
            }

            featureBatch = featureBatch.to(state->device);
            poseBatch = poseBatch.to(state->device);

            // encode through frozen AEs
            torch::Tensor featureLatent, poseLatent;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
                poseLatent = PoseAEEncode(state->poseAutoEncoder, poseBatch);
            }

            // predict and optimize
            state->singlePosePredictorOptimizer->zero_grad();
            torch::Tensor predicted = state->singlePosePredictor->forward(featureLatent);
            torch::Tensor loss = torch::mse_loss(predicted, poseLatent);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(state->singlePosePredictor->parameters(), 1.0);
            state->singlePosePredictorOptimizer->step();

            state->singlePosePredictorLoss = loss.item<float>();
            state->singlePosePredictorLossSmoothed = state->singlePosePredictorIterations == 0
                ? state->singlePosePredictorLoss
                : state->singlePosePredictorLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                    + state->singlePosePredictorLoss * LOSS_SMOOTHING_FACTOR;
            state->singlePosePredictorIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "SinglePosePredictor Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// latent flow matching training
// all upstream networks (AEs + predictor) are frozen
//---------------------------------------------------------

static inline void NetworkTrainFlowForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->latentFlowModel) return;
    if (!state->flowOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentLatentAveragePredictor) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const int segLatentDim = SEGMENT_AE_LATENT_DIM;
    const Clock::time_point start = Clock::now();

    // freeze everything upstream — we only train the
    // flow model, all the rest is locked in place
    state->featuresAutoEncoder->eval();
    state->segmentAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->latentFlowModel->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // grab random frames and build feature + segment batches on CPU,
            // same pattern as the other training loops
            torch::Tensor featureBatch = torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch = torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
                memcpy(sPtr + b * flatDim, src, (size_t)segFrames * pgDim * sizeof(float));
            }

            featureBatch = featureBatch.to(state->device);
            segmentBatch = segmentBatch.to(state->device);

            // run everything through the frozen networks to get latent representations
            torch::Tensor featureLatent;
            torch::Tensor trueSegLatent;
            torch::Tensor avgLatent;
            {
                torch::NoGradGuard noGrad;
                // features -> 16d latent
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
                // full pose segment -> 128d latent
                trueSegLatent = EncodeWithAE(state->segmentAutoEncoder, segmentBatch);
                // what the predictor thinks the average pose should be
                avgLatent = state->segmentLatentAveragePredictor->forward(featureLatent);
            }

#ifdef DO_DELTA_FLOWMATCHING
            // Delta flow matching: predict delta from average
            torch::Tensor delta = trueSegLatent - avgLatent;
            torch::Tensor x1 = delta;
#else
            // Direct flow matching: predict segment latent directly
            torch::Tensor x1 = trueSegLatent;
#endif



            // each sample gets its own random time in [0,1],
            // the "how far along the flow are we?" parameter
            torch::Tensor t = torch::rand(
                { batchSize, 1 }, torch::TensorOptions().device(state->device));

            // pure gaussian noise, the starting point of the flow at t=0
            torch::Tensor x0 = torch::randn(
                { batchSize, segLatentDim }, torch::TensorOptions().device(state->device));

            // linear interpolation between noise and the true target —
            // this is where we are at time t along the straight-line flow
            torch::Tensor xt = (1.0f - t) * x0 + t * x1;

#ifdef FLOW_PREDICT_X1
            // Endpoint prediction: network predicts x1 directly
            torch::Tensor target = x1;
#else
            // Velocity prediction: network predicts v = (x1 - xt) / (1 - t)
            // avoid division by zero at t=1
            torch::Tensor velocity = (x1 - xt) / (1.0f - t + 1e-8f);
            torch::Tensor target = velocity;
#endif

#ifdef DO_DELTA_FLOWMATCHING
            // the network sees: conditioning info (what motion features we want +
            // what the average looks like), the noisy sample xt, and time t
            torch::Tensor flowInput = torch::cat(
                { featureLatent, avgLatent, xt, t }, /*dim=*/1);
#else
            // the network sees: conditioning info (what motion features we want),
            // the noisy sample xt, and time t (no average conditioning)
            torch::Tensor flowInput = torch::cat(
                { featureLatent, xt, t }, /*dim=*/1);
#endif

            // endpoint prediction: the network tries to guess where the flow ends up
            state->flowOptimizer->zero_grad();
            torch::Tensor predicted = state->latentFlowModel->forward(flowInput);
            torch::Tensor loss = torch::mse_loss(predicted, target);
            loss.backward();
            state->flowOptimizer->step();

            state->flowLoss = loss.item<float>();
            state->flowLossSmoothed = state->flowLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                + state->flowLoss * LOSS_SMOOTHING_FACTOR;
            state->flowIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR,
            "Flow Training Error: %s", e.what());
        state->isTraining = false;
    }

    if (state->flowIterations % 300 == 0
        && state->flowIterations > 0)
    {
        TraceLog(LOG_INFO,
            "Flow Loss (iter %d): %.6f",
            state->flowIterations,
            state->flowLoss);
    }
}

//---------------------------------------------------------
// full-space flow matching training (no AEs, direct segment space)
//---------------------------------------------------------

static inline void NetworkTrainFullFlowForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->fullFlowModel) return;
    if (budgetSeconds <= 0.0) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const Clock::time_point start = Clock::now();

    state->fullFlowModel->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            // build condition and segment batches on CPU
            torch::Tensor condBatch = torch::empty({batchSize, featureDim});
            torch::Tensor x1Batch = torch::empty({batchSize, flatDim});
            float* cPtr = condBatch.data_ptr<float>();
            float* sPtr = x1Batch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), cPtr + b * featureDim);

                const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
                memcpy(sPtr + b * flatDim, src, (size_t)segFrames * pgDim * sizeof(float));
            }

            condBatch = condBatch.to(state->device);
            x1Batch = x1Batch.to(state->device);

            // sample time and noise
            torch::Tensor t = torch::rand(
                {batchSize, 1},
                torch::TensorOptions().device(state->device));
            torch::Tensor x0 = torch::randn(
                {batchSize, flatDim},
                torch::TensorOptions().device(state->device));

            // interpolate: xt = (1-t)*x0 + t*x1
            torch::Tensor xt = (1.0f - t) * x0 + t * x1Batch;

            // target velocity: x1 - x0 (constant along OT path)
            torch::Tensor target = x1Batch - x0;

            // condition + time concatenated for re-injection
            torch::Tensor condTime = torch::cat({condBatch, t}, 1);

            // forward and optimize
            state->fullFlowOptimizer->zero_grad();
            torch::Tensor predicted = state->fullFlowModel->forward(xt, condTime);
            torch::Tensor loss = torch::mse_loss(predicted, target);

            loss.backward();
            state->fullFlowOptimizer->step();

            state->fullFlowLoss = loss.item<float>();
            state->fullFlowLossSmoothed = state->fullFlowIterations == 0
                ? state->fullFlowLoss
                : state->fullFlowLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                  + state->fullFlowLoss * LOSS_SMOOTHING_FACTOR;
            state->fullFlowIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "FullFlow Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// friday flow training: flow matching predicting actual next latent
// (ControlOperators-style: x1 = z1, Zdist-scaled noise, noisy z0 conditioning)
//---------------------------------------------------------

static inline void NetworkTrainFridayFlowForTime(NetworkState* state, const AnimDatabase* db, double budgetSeconds)
{
    if (!state->fridayFlowModel) return;
    if (!state->poseAutoEncoder) return;
    if (!state->featuresAutoEncoder) return;
    if (budgetSeconds <= 0.0) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;
    if (state->poseLatentStd.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const int latentDim = POSE_AE_LATENT_DIM;
    const Clock::time_point start = Clock::now();

    state->fridayFlowModel->train();
    state->poseAutoEncoder->eval();
    state->featuresAutoEncoder->eval();

    // build Zdist tensor from stored per-dim std (kept on device for the loop)
    torch::Tensor Zdist = torch::from_blob(
        state->poseLatentStd.data(), {1, latentDim}, torch::kFloat32)
        .clone().to(state->device);

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor condBatch = torch::empty({batchSize, featureDim});
            torch::Tensor poseBatch0 = torch::empty({batchSize, poseDim});
            torch::Tensor poseBatch1 = torch::empty({batchSize, poseDim});
            float* cPtr = condBatch.data_ptr<float>();
            float* p0Ptr = poseBatch0.data_ptr<float>();
            float* p1Ptr = poseBatch1.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                // legal frames have enough buffer from clip boundaries
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), cPtr + b * featureDim);

                std::span<const float> pRow0 = db->normalizedPoseGenFeatures.row_view(frame);
                std::copy(pRow0.begin(), pRow0.end(), p0Ptr + b * poseDim);

                std::span<const float> pRow1 = db->normalizedPoseGenFeatures.row_view(frame + 1);
                std::copy(pRow1.begin(), pRow1.end(), p1Ptr + b * poseDim);
            }

            condBatch = condBatch.to(state->device);
            poseBatch0 = poseBatch0.to(state->device);
            poseBatch1 = poseBatch1.to(state->device);

            // encode everything through frozen AEs:
            // features -> feature latent (compact conditioning)
            // poses -> pose latent (what the flow predicts)
            torch::Tensor featureLatent, zPrev, zNext;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, condBatch);
                zPrev = PoseAEEncode(state->poseAutoEncoder, poseBatch0);
                zNext = PoseAEEncode(state->poseAutoEncoder, poseBatch1);
            }

            // flow matching noise: random starting point of the flow path,
            // scaled by per-dim data distribution so noise is closer to the actual
            // latent distribution (less transport distance for the flow to learn)
            torch::Tensor flowNoise = Zdist * torch::randn(
                {batchSize, latentDim}, torch::TensorOptions().device(state->device));

            // noisy conditioning: add noise to zPrev so the network learns to handle
            // imperfect input at runtime (where zPrev is a previous prediction, not a
            // clean AE encoding). Each sample gets a random noise level 0..POSE_NOISE.
            torch::Tensor zPrevNoisy = zPrev + Zdist * FRIDAY_FLOW_POSE_NOISE
                * torch::rand({batchSize, 1}, torch::TensorOptions().device(state->device))
                * torch::randn_like(zPrev);

            // flow matching: sample t
            torch::Tensor t = torch::rand(
                {batchSize, 1}, torch::TensorOptions().device(state->device));

            // interpolate along flow path: from noise toward the target next latent
            torch::Tensor zt = (1.0f - t) * flowNoise + t * zNext;

            // target: constant velocity along OT path (zNext - flowNoise)
            torch::Tensor target = zNext - flowNoise;

            // condition = cat(featureLatent, zPrevNoisy, t)
            torch::Tensor condTime = torch::cat({featureLatent, zPrevNoisy, t}, 1);

            // forward: network output scaled by Zdist (denormalization of velocity)
            state->fridayFlowOptimizer->zero_grad();
            torch::Tensor predicted = Zdist * state->fridayFlowModel->forward(zt, condTime);
            torch::Tensor loss = torch::mse_loss(predicted, target);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(state->fridayFlowModel->parameters(), 1.0);
            state->fridayFlowOptimizer->step();

            state->fridayFlowLoss = loss.item<float>();
            state->fridayFlowLossSmoothed = state->fridayFlowIterations == 0
                ? state->fridayFlowLoss
                : state->fridayFlowLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR) + state->fridayFlowLoss * LOSS_SMOOTHING_FACTOR;
            state->fridayFlowIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "FridayFlow Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// friday flow inference: predict the actual next latent
//---------------------------------------------------------

static inline bool NetworkPredictFridayFlow(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    const std::vector<float>& currentLatent,
    /*out*/ std::vector<float>& predictedNextLatent)
{
    if (!state->fridayFlowModel) return false;
    if (!state->poseAutoEncoder) return false;
    if (!state->featuresAutoEncoder) return false;
    if (state->poseLatentStd.empty()) return false;

    const int featureDim = db->featureDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->fridayFlowModel->eval();
    state->featuresAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // Zdist for scaling noise and network output
    torch::Tensor Zdist = torch::from_blob(
        state->poseLatentStd.data(), {1, latentDim}, torch::kFloat32)
        .clone().to(state->device);

    // normalize the raw MM query, then encode through feature AE
    torch::Tensor normQuery = torch::empty({1, featureDim});
    NormalizeFeatureQuery(db, rawQuery.data(), normQuery.data_ptr<float>());
    normQuery = normQuery.to(state->device);
    torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, normQuery);

    // current latent as tensor (this is zPrev — the conditioning)
    torch::Tensor zPrev = torch::from_blob(
        (void*)currentLatent.data(), {1, latentDim}, torch::kFloat32)
        .clone().to(state->device);

    // start from distribution-matched noise
    const int steps = FLOW_INFERENCE_STEPS;
    //torch::Tensor x = Zdist * torch::randn(
    //    { 1, latentDim }, torch::TensorOptions().device(state->device));
    // start from zero: deterministic at runtime
    torch::Tensor x = Zdist * torch::zeros(
        { 1, latentDim }, torch::TensorOptions().device(state->device));

    // Euler integration: flow from noise toward the predicted next latent
    for (int s = 0; s < steps; ++s)
    {
        const float tVal = (float)s / (float)steps;
        torch::Tensor t = torch::full(
            {1, 1}, tVal, torch::TensorOptions().device(state->device));
        torch::Tensor condTime = torch::cat({featureLatent, zPrev, t}, 1);
        torch::Tensor v = Zdist * state->fridayFlowModel->forward(x, condTime);
        x = x + v * (1.0f / steps);
    }

    // bail if the ODE produced NaN/Inf (untrained or unstable network)
    if (!x.isfinite().all().item<bool>())
    {
        return false;
    }

    // x is now the predicted next latent (not a delta)
    predictedNextLatent.resize(latentDim);
    torch::Tensor xCpu = x.to(torch::kCPU);
    const float* xPtr = xCpu.data_ptr<float>();
    for (int d = 0; d < latentDim; ++d)
    {
        predictedNextLatent[d] = xPtr[d];
    }
    return true;
}

//---------------------------------------------------------
// friday flow: decode latent to raw poseGenFeatures
//---------------------------------------------------------

static inline bool NetworkDecodeFridayFlowLatent(
    NetworkState* state, const AnimDatabase* db, const std::vector<float>& latent, /*out*/ std::vector<float>& rawPose)
{
    if (!state->poseAutoEncoder) return false;

    const int latentDim = POSE_AE_LATENT_DIM;
    const int poseDim = db->poseGenFeaturesComputeDim;

    state->poseAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    torch::Tensor z = torch::from_blob((void*)latent.data(), {1, latentDim}, torch::kFloat32).clone().to(state->device);

    torch::Tensor decoded = PoseAEDecode(state->poseAutoEncoder, z);
    decoded = decoded.to(torch::kCPU);
    const float* dPtr = decoded.data_ptr<float>();

    // denormalize: raw = (val / weight) * std + mean, then clamp to mocap bounds
    rawPose.resize(poseDim);
    for (int d = 0; d < poseDim; ++d)
    {
        const float w = db->poseGenFeaturesWeight[d];
        if (w > 1e-10f)
        {
            rawPose[d] = (dPtr[d] / w) * db->poseGenFeaturesStd[d] + db->poseGenFeaturesMean[d];
        }
        else
        {
            rawPose[d] = db->poseGenFeaturesMean[d];
        }
        rawPose[d] = std::clamp(rawPose[d], db->poseGenFeaturesMin[d], db->poseGenFeaturesMax[d]);
    }

    return true;
}

//---------------------------------------------------------
// friday flow: encode raw poseGenFeatures to latent
//---------------------------------------------------------

static inline bool NetworkEncodePoseToLatent(
    NetworkState* state, const AnimDatabase* db, const std::vector<float>& rawPose, /*out*/ std::vector<float>& latent)
{
    if (!state->poseAutoEncoder) return false;

    const int poseDim = db->poseGenFeaturesComputeDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->poseAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // normalize
    torch::Tensor input = torch::empty({1, poseDim});
    float* iPtr = input.data_ptr<float>();
    for (int d = 0; d < poseDim; ++d)
    {
        const float std = db->poseGenFeaturesStd[d];
        const float w = db->poseGenFeaturesWeight[d];
        if (std > 1e-10f)
        {
            iPtr[d] = (rawPose[d] - db->poseGenFeaturesMean[d]) / std * w;
        }
        else
        {
            iPtr[d] = 0.0f;
        }
    }
    input = input.to(state->device);

    torch::Tensor z = PoseAEEncode(state->poseAutoEncoder, input);
    z = z.to(torch::kCPU);
    const float* zPtr = z.data_ptr<float>();

    latent.resize(latentDim);
    for (int d = 0; d < latentDim; ++d) latent[d] = zPtr[d];

    return true;
}

//---------------------------------------------------------
// single pose predictor inference:
// features -> featureAE -> predict pose latent -> poseAE decode -> raw pose
//---------------------------------------------------------

static inline bool NetworkPredictSinglePose(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    /*out*/ std::vector<float>& rawPose)
{
    if (!state->singlePosePredictor) return false;
    if (!state->featuresAutoEncoder) return false;
    if (!state->poseAutoEncoder) return false;

    const int featureDim = db->featureDim;
    const int poseDim = db->poseGenFeaturesComputeDim;

    state->singlePosePredictor->eval();
    state->featuresAutoEncoder->eval();
    state->poseAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // normalize and encode features
    torch::Tensor normQuery = torch::empty({1, featureDim});
    NormalizeFeatureQuery(db, rawQuery.data(), normQuery.data_ptr<float>());
    normQuery = normQuery.to(state->device);
    torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, normQuery);

    // predict pose latent
    torch::Tensor poseLatent = state->singlePosePredictor->forward(featureLatent);

    // decode through PoseAE
    torch::Tensor decoded = PoseAEDecode(state->poseAutoEncoder, poseLatent);
    decoded = decoded.to(torch::kCPU);
    const float* dPtr = decoded.data_ptr<float>();

    // denormalize and clamp
    rawPose.resize(poseDim);
    for (int d = 0; d < poseDim; ++d)
    {
        const float w = db->poseGenFeaturesWeight[d];
        if (w > 1e-10f)
        {
            rawPose[d] = (dPtr[d] / w) * db->poseGenFeaturesStd[d] + db->poseGenFeaturesMean[d];
        }
        else
        {
            rawPose[d] = db->poseGenFeaturesMean[d];
        }
        rawPose[d] = std::clamp(rawPose[d], db->poseGenFeaturesMin[d], db->poseGenFeaturesMax[d]);
    }

    return true;
}

//---------------------------------------------------------
// single pose predictor: features → pose latent (no decode)
// used by UnconditionedAdvance mode to get conditioned latent on search frames
//---------------------------------------------------------

static inline bool NetworkPredictSinglePoseLatent(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    /*out*/ std::vector<float>& poseLatentOut)
{
    if (!state->singlePosePredictor) return false;
    if (!state->featuresAutoEncoder) return false;

    const int featureDim = db->featureDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->singlePosePredictor->eval();
    state->featuresAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // normalize and encode features
    torch::Tensor normQuery = torch::empty({1, featureDim});
    NormalizeFeatureQuery(db, rawQuery.data(), normQuery.data_ptr<float>());
    normQuery = normQuery.to(state->device);
    torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, normQuery);

    // predict pose latent (stop here, no PoseAE decode)
    torch::Tensor poseLatent = state->singlePosePredictor->forward(featureLatent);
    poseLatent = poseLatent.to(torch::kCPU);
    const float* zPtr = poseLatent.data_ptr<float>();

    poseLatentOut.resize(latentDim);
    for (int d = 0; d < latentDim; ++d)
    {
        poseLatentOut[d] = zPtr[d];
    }

    return true;
}

//---------------------------------------------------------
// unconditioned advance inference: pose latent → next pose latent
//---------------------------------------------------------

static inline bool NetworkPredictUncondAdvance(
    NetworkState* state,
    const std::vector<float>& currentLatent,
    /*out*/ std::vector<float>& predictedNextLatent)
{
    if (!state->uncondAdvancePredictor) return false;

    const int latentDim = POSE_AE_LATENT_DIM;

    state->uncondAdvancePredictor->eval();
    torch::NoGradGuard noGrad;

    torch::Tensor z = torch::from_blob(
        (void*)currentLatent.data(), {1, latentDim}, torch::kFloat32)
        .clone().to(state->device);

    torch::Tensor predicted = state->uncondAdvancePredictor->forward(z);

    if (!predicted.isfinite().all().item<bool>())
    {
        return false;
    }

    predicted = predicted.to(torch::kCPU);
    const float* pPtr = predicted.data_ptr<float>();
    predictedNextLatent.resize(latentDim);
    for (int d = 0; d < latentDim; ++d)
    {
        predictedNextLatent[d] = pPtr[d];
    }

    return true;
}

//---------------------------------------------------------
// monday predictor inference: (features, pose latent) → delta
//---------------------------------------------------------

static inline bool NetworkPredictMonday(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    const std::vector<float>& currentLatent,
    /*out*/ std::vector<float>& deltaOut)
{
    if (!state->mondayPredictor) return false;
    if (!state->featuresAutoEncoder) return false;

    const int featureDim = db->featureDim;
    const int latentDim = POSE_AE_LATENT_DIM;

    state->mondayPredictor->eval();
    state->featuresAutoEncoder->eval();
    torch::NoGradGuard noGrad;

    // normalize and encode features
    torch::Tensor normQuery = torch::empty({1, featureDim});
    NormalizeFeatureQuery(db, rawQuery.data(), normQuery.data_ptr<float>());
    normQuery = normQuery.to(state->device);
    torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, normQuery);

    // current pose latent
    torch::Tensor zCurrent = torch::from_blob(
        (void*)currentLatent.data(), {1, latentDim}, torch::kFloat32)
        .clone().to(state->device);

    // predict delta (network outputs in normalized space)
    torch::Tensor input = torch::cat({featureLatent, zCurrent}, 1);
    torch::Tensor delta = state->mondayPredictor->forward(input);

    // denormalize back to latent space scale
    if (!state->mondayDeltaMean.empty())
    {
        torch::Tensor deltaMean = torch::from_blob(
            (void*)state->mondayDeltaMean.data(), {1, latentDim}, torch::kFloat32)
            .clone().to(state->device);
        torch::Tensor deltaStd = torch::from_blob(
            (void*)state->mondayDeltaStd.data(), {1, latentDim}, torch::kFloat32)
            .clone().to(state->device);
        delta = delta * deltaStd + deltaMean;
    }

    if (!delta.isfinite().all().item<bool>())
    {
        return false;
    }

    delta = delta.to(torch::kCPU);
    const float* dPtr = delta.data_ptr<float>();
    deltaOut.resize(latentDim);
    for (int d = 0; d < latentDim; ++d)
    {
        deltaOut[d] = dPtr[d];
    }

    return true;
}

//---------------------------------------------------------
// segment end-to-end training: features → encode → predict → decode → segment
// gradients flow through all 3 networks simultaneously
//---------------------------------------------------------

static inline void NetworkTrainSegmentEndToEndForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->segmentAutoEncoder) return;
    if (!state->segmentLatentAveragePredictor) return;
    if (!state->featureAEOptimizer) return;
    if (!state->segmentOptimizer) return;
    if (!state->latentSegmentPredictorOptimizer) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;
    const Clock::time_point start = Clock::now();

    // everything trains together — no frozen networks here
    state->featuresAutoEncoder->train();
    state->segmentAutoEncoder->train();
    state->segmentLatentAveragePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor featureBatch = torch::empty({ batchSize, featureDim });
            torch::Tensor segmentBatch = torch::empty({ batchSize, flatDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* sPtr = segmentBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                const float* src = db->normalizedPoseGenFeatures.data() + frame * pgDim;
                memcpy(sPtr + b * flatDim, src, (size_t)segFrames * pgDim * sizeof(float));
            }

            featureBatch = featureBatch.to(state->device);
            segmentBatch = segmentBatch.to(state->device);

            // the full chain: encode features, predict segment latent, decode to segment
            // no NoGradGuard — gradients flow through everything
            torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
            torch::Tensor predictedSegLatent = state->segmentLatentAveragePredictor->forward(featureLatent);
            torch::Tensor reconstructed = DecodeWithAE(state->segmentAutoEncoder, predictedSegLatent);

            // loss in output space: the network chain must reconstruct the true segment
            // this is in normalized bone-weighted space, so important joints matter more
            state->featureAEOptimizer->zero_grad();
            state->segmentOptimizer->zero_grad();
            state->latentSegmentPredictorOptimizer->zero_grad();

            torch::Tensor loss = torch::mse_loss(reconstructed, segmentBatch);
            loss.backward();

            state->featureAEOptimizer->step();
            state->segmentOptimizer->step();
            state->latentSegmentPredictorOptimizer->step();

            const float lossVal = loss.item<float>();
            state->segmentE2eLoss = lossVal;
            state->segmentE2eLossSmoothed = state->segmentE2eIterations == 0
                ? lossVal
                : state->segmentE2eLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR) + lossVal * LOSS_SMOOTHING_FACTOR;
            state->segmentE2eIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Segment E2E Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// single pose end-to-end training: features → featureAE encode → predict → poseAE decode → pose
// gradients flow through all 3 networks simultaneously
//---------------------------------------------------------

static inline void NetworkTrainSinglePoseEndToEndForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->featuresAutoEncoder) return;
    if (!state->poseAutoEncoder) return;
    if (!state->singlePosePredictor) return;
    if (!state->featureAEOptimizer) return;
    if (!state->poseAEOptimizer) return;
    if (!state->singlePosePredictorOptimizer) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const Clock::time_point start = Clock::now();

    // everything trains together — no frozen networks here
    state->featuresAutoEncoder->train();
    state->poseAutoEncoder->train();
    state->singlePosePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor featureBatch = torch::empty({ batchSize, featureDim });
            torch::Tensor poseBatch = torch::empty({ batchSize, poseDim });
            float* fPtr = featureBatch.data_ptr<float>();
            float* pPtr = poseBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                std::span<const float> pRow = db->normalizedPoseGenFeatures.row_view(frame);
                std::copy(pRow.begin(), pRow.end(), pPtr + b * poseDim);
            }

            featureBatch = featureBatch.to(state->device);
            poseBatch = poseBatch.to(state->device);

            // full chain: encode features, predict pose latent, decode to pose
            torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
            torch::Tensor predictedPoseLatent = state->singlePosePredictor->forward(featureLatent);
            torch::Tensor reconstructed = PoseAEDecode(state->poseAutoEncoder, predictedPoseLatent);

            // loss in output space: the network chain must reconstruct the true pose
            state->featureAEOptimizer->zero_grad();
            state->poseAEOptimizer->zero_grad();
            state->singlePosePredictorOptimizer->zero_grad();

            torch::Tensor loss = torch::mse_loss(reconstructed, poseBatch);
            loss.backward();

            state->featureAEOptimizer->step();
            state->poseAEOptimizer->step();
            state->singlePosePredictorOptimizer->step();

            const float lossVal = loss.item<float>();
            state->singlePoseE2eLoss = lossVal;
            state->singlePoseE2eLossSmoothed = state->singlePoseE2eIterations == 0
                ? lossVal
                : state->singlePoseE2eLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                    + lossVal * LOSS_SMOOTHING_FACTOR;
            state->singlePoseE2eIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "SinglePose E2E Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// unconditioned pose advance training: pose latent → next pose latent
// PoseAE is frozen, only the advance predictor trains
//---------------------------------------------------------

static inline void NetworkTrainUncondAdvanceForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->uncondAdvancePredictor) return;
    if (!state->uncondAdvanceOptimizer) return;
    if (!state->poseAutoEncoder) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const Clock::time_point start = Clock::now();

    state->poseAutoEncoder->eval();
    state->uncondAdvancePredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor poseBatch0 = torch::empty({batchSize, poseDim});
            torch::Tensor poseBatch1 = torch::empty({batchSize, poseDim});
            float* p0Ptr = poseBatch0.data_ptr<float>();
            float* p1Ptr = poseBatch1.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> pRow0 = db->normalizedPoseGenFeatures.row_view(frame);
                std::copy(pRow0.begin(), pRow0.end(), p0Ptr + b * poseDim);

                std::span<const float> pRow1 = db->normalizedPoseGenFeatures.row_view(frame + 1);
                std::copy(pRow1.begin(), pRow1.end(), p1Ptr + b * poseDim);
            }

            poseBatch0 = poseBatch0.to(state->device);
            poseBatch1 = poseBatch1.to(state->device);

            // encode through frozen PoseAE
            torch::Tensor zCurrent, zNext;
            {
                torch::NoGradGuard noGrad;
                zCurrent = PoseAEEncode(state->poseAutoEncoder, poseBatch0);
                zNext = PoseAEEncode(state->poseAutoEncoder, poseBatch1);
            }

            // add noise to input for robustness (at runtime the input is a previous prediction,
            // not a clean AE encoding, so the network needs to handle imperfect inputs)
            torch::Tensor noise = torch::randn_like(zCurrent) * 0.1f;
            torch::Tensor zCurrentNoisy = zCurrent + noise;

            // predict and optimize
            state->uncondAdvanceOptimizer->zero_grad();
            torch::Tensor predicted = state->uncondAdvancePredictor->forward(zCurrentNoisy);
            torch::Tensor loss = torch::mse_loss(predicted, zNext);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(state->uncondAdvancePredictor->parameters(), 1.0);
            state->uncondAdvanceOptimizer->step();

            state->uncondAdvanceLoss = loss.item<float>();
            state->uncondAdvanceLossSmoothed = state->uncondAdvanceIterations == 0
                ? state->uncondAdvanceLoss
                : state->uncondAdvanceLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                    + state->uncondAdvanceLoss * LOSS_SMOOTHING_FACTOR;
            state->uncondAdvanceIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "UncondAdvance Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// monday predictor training: (features, pose latent) → pose latent delta
// both AEs frozen, only mondayPredictor trains
//---------------------------------------------------------

static inline void NetworkTrainMondayForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->mondayPredictor) return;
    if (!state->mondayOptimizer) return;
    if (!state->featuresAutoEncoder) return;
    if (!state->poseAutoEncoder) return;
    if (db->normalizedFeatures.empty()) return;
    if (db->normalizedPoseGenFeatures.empty()) return;

    const int batchSize = 64;
    const int featureDim = db->featureDim;
    const int poseDim = db->poseGenFeaturesComputeDim;
    const int latentDim = POSE_AE_LATENT_DIM;
    const Clock::time_point start = Clock::now();

    // build normalization tensors if stats are available
    const bool hasStats = !state->mondayDeltaMean.empty();
    torch::Tensor deltaMean;
    torch::Tensor deltaStd;
    if (hasStats)
    {
        deltaMean = torch::from_blob(
            (void*)state->mondayDeltaMean.data(), {1, latentDim}, torch::kFloat32)
            .clone().to(state->device);
        deltaStd = torch::from_blob(
            (void*)state->mondayDeltaStd.data(), {1, latentDim}, torch::kFloat32)
            .clone().to(state->device);
    }

    state->featuresAutoEncoder->eval();
    state->poseAutoEncoder->eval();
    state->mondayPredictor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor featureBatch = torch::empty({batchSize, featureDim});
            torch::Tensor poseBatch0 = torch::empty({batchSize, poseDim});
            torch::Tensor poseBatch1 = torch::empty({batchSize, poseDim});
            float* fPtr = featureBatch.data_ptr<float>();
            float* p0Ptr = poseBatch0.data_ptr<float>();
            float* p1Ptr = poseBatch1.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalFrame(db);

                std::span<const float> fRow = db->normalizedFeatures.row_view(frame);
                std::copy(fRow.begin(), fRow.end(), fPtr + b * featureDim);

                std::span<const float> pRow0 = db->normalizedPoseGenFeatures.row_view(frame);
                std::copy(pRow0.begin(), pRow0.end(), p0Ptr + b * poseDim);

                std::span<const float> pRow1 = db->normalizedPoseGenFeatures.row_view(frame + 1);
                std::copy(pRow1.begin(), pRow1.end(), p1Ptr + b * poseDim);
            }

            featureBatch = featureBatch.to(state->device);
            poseBatch0 = poseBatch0.to(state->device);
            poseBatch1 = poseBatch1.to(state->device);

            // encode through frozen AEs
            torch::Tensor featureLatent, zCurrent, zNext;
            {
                torch::NoGradGuard noGrad;
                featureLatent = EncodeWithAE(state->featuresAutoEncoder, featureBatch);
                zCurrent = PoseAEEncode(state->poseAutoEncoder, poseBatch0);
                zNext = PoseAEEncode(state->poseAutoEncoder, poseBatch1);
            }

            // add noise to input pose latent for robustness.
            // the target delta adjusts accordingly: targetDelta = zNext - zCurrentNoisy
            // which equals (cleanDelta - noise), teaching the network to correct back
            // towards the trajectory when given drifted input at runtime
            torch::Tensor noise = torch::randn_like(zCurrent) * 0.1f;
            torch::Tensor zCurrentNoisy = zCurrent + noise;
            torch::Tensor targetDelta = zNext - zCurrentNoisy;

            // normalize target so the network outputs something close to gaussian
            if (hasStats)
            {
                targetDelta = (targetDelta - deltaMean) / deltaStd;
            }

            // predict delta from (featureLatent, noisyPoseLatent)
            torch::Tensor input = torch::cat({featureLatent, zCurrentNoisy}, 1);

            state->mondayOptimizer->zero_grad();
            torch::Tensor predictedDelta = state->mondayPredictor->forward(input);
            torch::Tensor loss = torch::mse_loss(predictedDelta, targetDelta);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(state->mondayPredictor->parameters(), 1.0);
            state->mondayOptimizer->step();

            state->mondayLoss = loss.item<float>();
            state->mondayLossSmoothed = state->mondayIterations == 0
                ? state->mondayLoss
                : state->mondayLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                    + state->mondayLoss * LOSS_SMOOTHING_FACTOR;
            state->mondayIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Monday Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// unified training: all networks, time-budgeted
//---------------------------------------------------------

// returns the wall time spent training (seconds)
static inline double NetworkTrainAll(
    NetworkState* state,
    const AnimDatabase* db,
    double totalBudgetSeconds)
{
    if (!state->isTraining || !db->valid) return 0.0;
    if (!state->featuresAutoEncoder) return 0.0;
    if (!state->segmentAutoEncoder) return 0.0;
    if (db->legalStartFrames.empty()) return 0.0;

    const Clock::time_point wallStart = Clock::now();

    const double elapsed = state->trainingElapsedSeconds;
    const bool hasLatentSegmentPredictor =
        static_cast<bool>(state->segmentLatentAveragePredictor);
    const bool hasFlow =
        static_cast<bool>(state->latentFlowModel);
    const bool hasFridayFlow =
        static_cast<bool>(state->fridayFlowModel);
    const bool hasSinglePosePredictor =
        static_cast<bool>(state->singlePosePredictor);
    const bool hasUncondAdvance =
        static_cast<bool>(state->uncondAdvancePredictor);
    const bool hasMonday =
        static_cast<bool>(state->mondayPredictor);

    // auto-init fullFlow immediately (no headstart needed, doesn't use AEs)
    if (!state->fullFlowModel &&
        db->featureDim > 0 && db->poseGenSegmentFlatDim > 0)
    {
        NetworkInitFullFlow(state, db->featureDim, db->poseGenSegmentFlatDim);
    }

    const bool hasFullFlow =
        static_cast<bool>(state->latentFlowModel);


    // split otherBudget among existing networks using relative weights
    // phase 1 (0-5s): AEs only. phase 2 (5s+): everything
    const float wFeature = BUDGET_WEIGHT_FEATURE_AE;
    const float wPose = BUDGET_WEIGHT_POSE_AE;
    const float wSegment = BUDGET_WEIGHT_SEGMENT_AE;
    const float wPredictor = hasLatentSegmentPredictor ? BUDGET_WEIGHT_LATENT_SEGMENT_PREDICTOR : 0.0f;
    const float wSegmentE2e = hasLatentSegmentPredictor ? BUDGET_WEIGHT_SEGMENT_E2E : 0.0f;
    const float wFlow = hasFlow ? BUDGET_WEIGHT_SEGMENT_FLOW : 0.0f;
    const float wFullFlow = hasFlow ? BUDGET_WEIGHT_FULL_FLOW : 0.0f;
    const float wFridayFlow = hasFridayFlow ? BUDGET_WEIGHT_FRIDAY_FLOW : 0.0f;
    const float wSinglePose = hasSinglePosePredictor ? BUDGET_WEIGHT_SINGLE_POSE_PREDICTOR : 0.0f;
    const float wSinglePoseE2e = hasSinglePosePredictor ? BUDGET_WEIGHT_POSE_E2E : 0.0f;
    const float wUncondAdvance = hasUncondAdvance ? BUDGET_WEIGHT_UNCOND_ADVANCE : 0.0f;
    const float wMonday = hasMonday ? BUDGET_WEIGHT_MONDAY : 0.0f;
    const float totalWeight = wFeature + wPose + wSegment + wPredictor + wSegmentE2e
        + wFlow + wFullFlow + wFridayFlow + wSinglePose + wSinglePoseE2e
        + wUncondAdvance + wMonday;

    const double featureBudget = totalBudgetSeconds * wFeature / totalWeight;
    const double poseBudget = totalBudgetSeconds * wPose / totalWeight;
    const double segmentBudget = totalBudgetSeconds * wSegment / totalWeight;
    const double predictorBudget = totalBudgetSeconds * wPredictor / totalWeight;
    const double segmentE2eBudget = totalBudgetSeconds * wSegmentE2e / totalWeight;
    const double flowBudget = totalBudgetSeconds * wFlow / totalWeight;
    const double fullFlowBudget = totalBudgetSeconds * wFullFlow / totalWeight;
    const double fridayFlowBudget = totalBudgetSeconds * wFridayFlow / totalWeight;
    const double singlePoseBudget = totalBudgetSeconds * wSinglePose / totalWeight;
    const double singlePoseE2eBudget = totalBudgetSeconds * wSinglePoseE2e / totalWeight;
    const double uncondAdvanceBudget = totalBudgetSeconds * wUncondAdvance / totalWeight;
    const double mondayBudget = totalBudgetSeconds * wMonday / totalWeight;

    // train fullFlow first (gets the most budget)
    if (hasFullFlow)
        NetworkTrainFullFlowForTime(state, db, fullFlowBudget);

    NetworkTrainFeatureAEForTime(state, db, featureBudget);
    NetworkTrainPoseAEForTime(state, db, poseBudget);
    NetworkTrainSegmentAEForTime(state, db, segmentBudget);

    // auto-init predictor and latent flow after AE headstart
    if (!state->segmentLatentAveragePredictor && elapsed >= HEADSTART_SECONDS)
    {
        NetworkInitLatentSegmentPredictor(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM);
        TraceLog(LOG_INFO, "AE headstart done, predictor initialized.");
    }

    if (!state->latentFlowModel && elapsed >= HEADSTART_SECONDS)
    {
        NetworkInitFlow(state, FEATURE_AE_LATENT_DIM, SEGMENT_AE_LATENT_DIM);
        TraceLog(LOG_INFO, "AE headstart done, flow model initialized.");
    }

    if (state->segmentLatentAveragePredictor && predictorBudget > 0.0)
        NetworkTrainLatentSegmentPredictorForTime(state, db, predictorBudget);

    if (state->segmentLatentAveragePredictor && segmentE2eBudget > 0.0)
        NetworkTrainSegmentEndToEndForTime(state, db, segmentE2eBudget);

    if (state->latentFlowModel && flowBudget > 0.0)
        NetworkTrainFlowForTime(state, db, flowBudget);

    // auto-init fridayFlow after PoseAE headstart
    // needs both PoseAE and FeatureAE since we condition on feature latents
    if (!state->fridayFlowModel
        && elapsed >= FRIDAY_FLOW_HEADSTART
        && state->poseAutoEncoder
        && state->featuresAutoEncoder
        && db->featureDim > 0)
    {
        NetworkInitFridayFlow(
            state, FEATURE_AE_LATENT_DIM,
            POSE_AE_LATENT_DIM);
        ComputePoseLatentStats(state, db);
        TraceLog(LOG_INFO,
            "PoseAE headstart done, "
            "FridayFlow initialized.");
    }

    // auto-init singlePosePredictor after headstart (same timing as FridayFlow)
    if (!state->singlePosePredictor
        && elapsed >= FRIDAY_FLOW_HEADSTART
        && state->poseAutoEncoder
        && state->featuresAutoEncoder)
    {
        NetworkInitSinglePosePredictor(state, FEATURE_AE_LATENT_DIM, POSE_AE_LATENT_DIM);
    }

    // auto-init uncondAdvance after headstart (needs PoseAE)
    if (!state->uncondAdvancePredictor
        && elapsed >= FRIDAY_FLOW_HEADSTART
        && state->poseAutoEncoder)
    {
        NetworkInitUncondAdvance(state, POSE_AE_LATENT_DIM);
    }

    // auto-init monday after headstart (needs both AEs)
    if (!state->mondayPredictor
        && elapsed >= FRIDAY_FLOW_HEADSTART
        && state->poseAutoEncoder
        && state->featuresAutoEncoder)
    {
        NetworkInitMonday(state, FEATURE_AE_LATENT_DIM, POSE_AE_LATENT_DIM);
        ComputeMondayDeltaStats(state, db);
    }

    // train fridayFlow (50% of total budget when active)
    if (state->fridayFlowModel)
    {
        NetworkTrainFridayFlowForTime(
            state, db, fridayFlowBudget);
    }

    if (state->singlePosePredictor && singlePoseBudget > 0.0)
    {
        NetworkTrainSinglePosePredictorForTime(state, db, singlePoseBudget);
    }

    if (state->singlePosePredictor && singlePoseE2eBudget > 0.0)
        NetworkTrainSinglePoseEndToEndForTime(state, db, singlePoseE2eBudget);

    if (state->uncondAdvancePredictor && uncondAdvanceBudget > 0.0)
        NetworkTrainUncondAdvanceForTime(state, db, uncondAdvanceBudget);

    if (state->mondayPredictor && mondayBudget > 0.0)
    {
        NetworkTrainMondayForTime(state, db, mondayBudget);
    }

    // periodically recompute monday delta stats (PoseAE latent space drifts as it trains)
    if (state->mondayPredictor && state->poseAutoEncoder)
    {
        state->mondayDeltaStatsTimer += totalBudgetSeconds;
        if (state->mondayDeltaStatsTimer >= MONDAY_DELTA_STATS_INTERVAL)
        {
            state->mondayDeltaStatsTimer = 0.0;
            ComputeMondayDeltaStats(state, db);
        }
    }

    const double wallElapsed = ElapsedSeconds(wallStart);
    state->trainingElapsedSeconds += wallElapsed;
    state->timeSinceLastAutoSave += wallElapsed;
    state->timeSinceLastLossLog += wallElapsed;

    if (state->timeSinceLastLossLog >= LOSS_LOG_INTERVAL_SECONDS)
    {
        state->timeSinceLastLossLog = 0.0;
        state->lossHistoryTime.push_back((float)state->trainingElapsedSeconds);
        state->featureAELossHistory.push_back(state->featureAEIterations >= 100 ? state->featureAELossSmoothed : 0.0f);
        state->poseAELossHistory.push_back(state->poseAEIterations >= 100 ? state->poseAELossSmoothed : 0.0f);
        state->segmentAELossHistory.push_back(state->segmentAEIterations >= 100 ? state->segmentAELossSmoothed : 0.0f);
        state->latentSegmentPredictorLossHistory.push_back(state->latentSegmentPredictorIterations >= 100 ? state->latentSegmentPredictorLossSmoothed : 0.0f);
        state->flowLossHistory.push_back(state->flowIterations >= 100 ? state->flowLossSmoothed : 0.0f);
        state->segmentE2eLossHistory.push_back(state->segmentE2eIterations >= 100 ? state->segmentE2eLossSmoothed : 0.0f);
        state->fullFlowLossHistory.push_back(state->fullFlowIterations >= 100 ? state->fullFlowLossSmoothed : 0.0f);
        state->fridayFlowLossHistory.push_back(state->fridayFlowIterations >= 100 ? state->fridayFlowLossSmoothed : 0.0f);
        state->singlePosePredictorLossHistory.push_back(state->singlePosePredictorIterations >= 100 ? state->singlePosePredictorLossSmoothed : 0.0f);
        state->singlePoseE2eLossHistory.push_back(state->singlePoseE2eIterations >= 100 ? state->singlePoseE2eLossSmoothed : 0.0f);
        state->uncondAdvanceLossHistory.push_back(state->uncondAdvanceIterations >= 100 ? state->uncondAdvanceLossSmoothed : 0.0f);
        state->mondayLossHistory.push_back(state->mondayIterations >= 100 ? state->mondayLossSmoothed : 0.0f);
    }
    return wallElapsed;
}

//---------------------------------------------------------
// init all networks for a fresh training session
//---------------------------------------------------------

static inline void NetworkInitAllForTraining(
    NetworkState* state,
    AnimDatabase* db)
{
    if (!db->valid) return;

    // cluster features for stratified training sampling (lazy, only once)
    if (db->clusterCount == 0)
        AnimDatabaseClusterFeatures(db);

    NetworkInitFeatureAE(
        state, db->featureDim, FEATURE_AE_LATENT_DIM);
    NetworkInitPoseAE(
        state, db->poseGenFeaturesComputeDim, POSE_AE_LATENT_DIM);

    if (db->poseGenSegmentFlatDim > 0)
    {
        NetworkInitSegmentAE(
            state, db->poseGenSegmentFlatDim,
            SEGMENT_AE_LATENT_DIM);
    }

    // predictor is NOT initialized here,
    // it auto-inits after the headstart period

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;
    state->isTraining = true;

    TraceLog(LOG_INFO,
        "All networks initialized. "
        "Predictor + Flow at %.0fs headstart.",
        HEADSTART_SECONDS);
}

// reset all networks to null (e.g. after database rebuild
// which invalidates feature dimensions)
static inline void NetworkResetAll(NetworkState* state)
{
    state->isTraining = false;

    state->featuresAutoEncoder = nullptr;
    state->featureAEOptimizer = nullptr;
    state->featureAELoss = 0.0f;
    state->featureAEIterations = 0;

    state->poseAutoEncoder = nullptr;
    state->poseAEOptimizer = nullptr;
    state->poseAELoss = 0.0f;
    state->poseAEIterations = 0;
    
    state->segmentAutoEncoder = nullptr;
    state->segmentOptimizer = nullptr;
    state->segmentAELoss = 0.0f;
    state->segmentAEIterations = 0;

    state->segmentLatentAveragePredictor = nullptr;
    state->latentSegmentPredictorOptimizer = nullptr;
    state->latentSegmentPredictorLoss = 0.0f;
    state->latentSegmentPredictorIterations = 0;

    state->singlePosePredictor = nullptr;
    state->singlePosePredictorOptimizer = nullptr;
    state->singlePosePredictorLoss = 0.0f;
    state->singlePosePredictorIterations = 0;

    state->latentFlowModel = nullptr;
    state->flowOptimizer = nullptr;
    state->flowLoss = 0.0f;
    state->flowIterations = 0;

    state->fullFlowModel = nullptr;
    state->fullFlowOptimizer = nullptr;
    state->fullFlowLoss = 0.0f;
    state->fullFlowLossSmoothed = 0.0f;
    state->fullFlowIterations = 0;

    state->fridayFlowModel = nullptr;
    state->fridayFlowOptimizer = nullptr;
    state->fridayFlowLoss = 0.0f;
    state->fridayFlowLossSmoothed = 0.0f;
    state->fridayFlowIterations = 0;
    state->poseLatentStd.clear();

    state->segmentE2eLoss = 0.0f;
    state->segmentE2eLossSmoothed = 0.0f;
    state->segmentE2eIterations = 0;

    state->singlePoseE2eLoss = 0.0f;
    state->singlePoseE2eLossSmoothed = 0.0f;
    state->singlePoseE2eIterations = 0;

    state->uncondAdvancePredictor = nullptr;
    state->uncondAdvanceOptimizer = nullptr;
    state->uncondAdvanceLoss = 0.0f;
    state->uncondAdvanceLossSmoothed = 0.0f;
    state->uncondAdvanceIterations = 0;

    state->mondayPredictor = nullptr;
    state->mondayOptimizer = nullptr;
    state->mondayLoss = 0.0f;
    state->mondayLossSmoothed = 0.0f;
    state->mondayIterations = 0;
    state->mondayDeltaMean.clear();
    state->mondayDeltaStd.clear();
    state->mondayDeltaStatsTimer = 0.0;

    state->lossHistoryTime.clear();
    state->featureAELossHistory.clear();
    state->poseAELossHistory.clear();
    state->segmentAELossHistory.clear();
    state->latentSegmentPredictorLossHistory.clear();
    state->singlePosePredictorLossHistory.clear();
    state->flowLossHistory.clear();
    state->segmentE2eLossHistory.clear();
    state->fullFlowLossHistory.clear();
    state->fridayFlowLossHistory.clear();
    state->singlePoseE2eLossHistory.clear();
    state->uncondAdvanceLossHistory.clear();
    state->mondayLossHistory.clear();
    state->timeSinceLastLossLog = 0.0;

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;

    TraceLog(LOG_INFO, "All networks reset.");
}

//---------------------------------------------------------
// predict a full segment from raw motion features
//---------------------------------------------------------

// features -> normalize -> encode -> predict latent ->
// decode -> denormalize -> segment of poseGenFeatures.
// Returns false if any required network is null.
static inline bool NetworkPredictSegment(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment)
{
    if (!state->featuresAutoEncoder) return false;
    if (!state->segmentLatentAveragePredictor) return false;
    if (!state->segmentAutoEncoder) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;

    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;

    segment.resize(segFrames, pgDim);

    // normalize query
    torch::Tensor queryTensor = torch::empty({ 1, featureDim });
    NormalizeFeatureQuery(db, rawQuery.data(), queryTensor.data_ptr<float>());

    state->featuresAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->segmentAutoEncoder->eval();

    queryTensor = queryTensor.to(state->device);

    torch::Tensor flat;
    {
        torch::NoGradGuard noGrad;

        // encode features -> feature latent
        torch::Tensor featureLatent = EncodeWithAE(
            state->featuresAutoEncoder, queryTensor);

        // predict -> segment latent
        torch::Tensor segmentLatent =
            state->segmentLatentAveragePredictor
                ->forward(featureLatent);

        // decode -> flat normalized poseGen
        flat = DecodeWithAE(
            state->segmentAutoEncoder, segmentLatent);
    }

    flat = flat.to(torch::kCPU);
    const float* fPtr = flat.data_ptr<float>();

    // denormalize and write into segment rows
    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w =
                db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (fPtr[f * pgDim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }

    return true;
}

//---------------------------------------------------------
// flow matching inference
// features -> encode -> predict average -> sample flow
// -> decode -> denormalize -> segment
//---------------------------------------------------------

static inline bool NetworkPredictSegmentFlow(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment)
{
    // need all 4 networks for this
    if (!state->featuresAutoEncoder) return false;
    if (!state->segmentLatentAveragePredictor) return false;
    if (!state->segmentAutoEncoder) return false;
    if (!state->latentFlowModel) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;

    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int segLatentDim = SEGMENT_AE_LATENT_DIM;

    segment.resize(segFrames, pgDim);

    // normalize the raw motion matching query
    torch::Tensor queryTensor = torch::empty({ 1, featureDim });
    NormalizeFeatureQuery(db, rawQuery.data(), queryTensor.data_ptr<float>());

    // everything is inference-only, no gradients needed
    state->featuresAutoEncoder->eval();
    state->segmentLatentAveragePredictor->eval();
    state->latentFlowModel->eval();
    state->segmentAutoEncoder->eval();

    queryTensor = queryTensor.to(state->device);

    torch::Tensor flat;
    {
        torch::NoGradGuard noGrad;

        // compress features down to a 16d latent
        torch::Tensor featureLatent = EncodeWithAE(state->featuresAutoEncoder, queryTensor);

        // the predictor gives us the "average" pose for these features — good but boring
        torch::Tensor avgLatent = state->segmentLatentAveragePredictor->forward(featureLatent);

        // this is where the magic happens: we start from pure noise and let the flow
        // model gradually shape it into a plausible result

        // mode seek from zero: the mode is better than random noise - crisp, hopefully
        torch::Tensor x0 = torch::zeros(
            { 1, segLatentDim }, torch::TensorOptions().device(state->device));
        //torch::Tensor x0 = torch::randn(
        //    { 1, segLatentDim }, torch::TensorOptions().device(state->device));

        // euler integration with endpoint prediction: at each step the network predicts
        // where x1 should be, and we re-interpolate from our original noise x0 towards
        // that prediction. more steps = smoother trajectory through latent space
        torch::Tensor x = x0;
        torch::Tensor x1Hat;
        for (int step = 0; step < FLOW_INFERENCE_STEPS; ++step)
        {
            const float t = (float)step / FLOW_INFERENCE_STEPS;
            torch::Tensor tTensor = torch::full(
                { 1, 1 }, t, torch::TensorOptions().device(state->device));

#ifdef DO_DELTA_FLOWMATCHING
            // condition on: what features we want, what the average looks like,
            // where we currently are in the flow, and how far along we are (t)
            torch::Tensor flowInput = torch::cat(
                { featureLatent, avgLatent, x, tTensor }, /*dim=*/1);
#else
            // condition on: what features we want, where we currently are in the flow,
            // and how far along we are (t) - no average conditioning
            torch::Tensor flowInput = torch::cat(
                { featureLatent, x, tTensor }, /*dim=*/1);
#endif

#ifdef FLOW_PREDICT_X1
            // the network predicts the endpoint x1
            x1Hat = state->latentFlowModel->forward(flowInput);

            // jump to the interpolated position between our starting noise
            // and the predicted endpoint at the next timestep
            const float nextT = (float)(step + 1) / FLOW_INFERENCE_STEPS;
            x = (1.0f - nextT) * x0 + nextT * x1Hat;
#else
            // the network predicts the velocity field v
            torch::Tensor v = state->latentFlowModel->forward(flowInput);

            // integrate: x += v * dt
            const float dt = 1.0f / FLOW_INFERENCE_STEPS;
            x = x + v * dt;

            // cache the last prediction as x1Hat for final result
            if (step == FLOW_INFERENCE_STEPS - 1)
            {
                x1Hat = x;
            }
#endif
        }

#ifdef DO_DELTA_FLOWMATCHING
        // the flow gave us a delta — add it back to the average
        // to get a diverse but plausible segment latent
        torch::Tensor segmentLatent = avgLatent + x1Hat;
#else
        // the flow gave us the segment latent directly
        torch::Tensor segmentLatent = x1Hat;
#endif
        // decode back from 128d latent to the full flat pose segment
        flat = DecodeWithAE(state->segmentAutoEncoder, segmentLatent);
    }

    // bring it back to CPU for the denorm loop
    flat = flat.to(torch::kCPU);
    const float* fPtr = flat.data_ptr<float>();

    // undo the normalization so we get real-world pose values back
    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w = db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (fPtr[f * pgDim + d] / w * db->poseGenFeaturesStd[d] + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }

    return true;
}

//---------------------------------------------------------
// full-space flow predict (no AEs, direct segment space)
//---------------------------------------------------------

static inline bool NetworkPredictFullFlow(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment)
{
    if (!state->fullFlowModel) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;

    const int featureDim = db->featureDim;
    const int pgDim = db->poseGenFeaturesComputeDim;
    const int segFrames = db->poseGenSegmentFrameCount;
    const int flatDim = db->poseGenSegmentFlatDim;

    segment.resize(segFrames, pgDim);

    // normalize the raw query
    torch::Tensor condTensor = torch::empty({1, featureDim});
    NormalizeFeatureQuery(db, rawQuery.data(), condTensor.data_ptr<float>());

    state->fullFlowModel->eval();
    condTensor = condTensor.to(state->device);

    torch::Tensor result;
    {
        torch::NoGradGuard noGrad;

        // start from noise
        //torch::Tensor x = torch::randn(
        //    {1, flatDim},
        //    torch::TensorOptions().device(state->device));
        // mode seek from zero: the mode is better than random noise - crisp, hopefully
        torch::Tensor x = torch::zeros(
            { 1, flatDim }, torch::TensorOptions().device(state->device));

        // Euler integration: 8 steps
        constexpr int steps = FLOW_INFERENCE_STEPS;
        constexpr float dt = 1.0f / steps;

        for (int step = 0; step < steps; ++step)
        {
            const float t = (float)step / steps;
            torch::Tensor tTensor = torch::full(
                {1, 1}, t,
                torch::TensorOptions().device(state->device));
            torch::Tensor condTime = torch::cat(
                {condTensor, tTensor}, 1);
            torch::Tensor v = state->fullFlowModel->forward(
                x, condTime);
            x = x + v * dt;
        }

        result = x.to(torch::kCPU);
    }

    // denormalize to get real-world pose values
    const float* fPtr = result.data_ptr<float>();
    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w = db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (fPtr[f * pgDim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }

    return true;
}

//---------------------------------------------------------
// segment AE apply (for visual evaluation)
//---------------------------------------------------------

// Pass a segment through the segment autoencoder
// in-place (normalize -> encode -> decode -> denormalize).
// Used to visually evaluate reconstruction quality.
static inline void NetworkApplySegmentAE(
    NetworkState* networkState,
    const AnimDatabase* db,
    Array2D<float>* segment)
{
    if (!networkState->segmentAutoEncoder) return;

    const int segFrameCount = segment->rows();
    const int dim = segment->cols();
    const int flatDim = segFrameCount * dim;

    // normalize: (raw - mean) / std * weight
    torch::Tensor normalized =
        torch::empty({ 1, flatDim });
    float* nPtr = normalized.data_ptr<float>();
    for (int f = 0; f < segFrameCount; ++f)
    {
        std::span<const float> row =
            segment->row_view(f);
        for (int d = 0; d < dim; ++d)
        {
            nPtr[f * dim + d] =
                (row[d] - db->poseGenFeaturesMean[d])
                / db->poseGenFeaturesStd[d]
                * db->poseGenFeaturesWeight[d];
        }
    }

    assertEvenInRelease(
        db->poseGenSegmentFlatDim == flatDim);

    // forward pass (eval mode)
    normalized = normalized.to(networkState->device);
    networkState->segmentAutoEncoder->eval();
    torch::Tensor reconstructed =
        networkState->segmentAutoEncoder
            ->forward(normalized);
    reconstructed = reconstructed.to(torch::kCPU);

    // denormalize: raw = output / weight * std + mean
    const float* rPtr = reconstructed.data_ptr<float>();
    for (int f = 0; f < segFrameCount; ++f)
    {
        std::span<float> dst = segment->row_view(f);
        for (int d = 0; d < dim; ++d)
        {
            const float w =
                db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (rPtr[f * dim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }
}


// Pass a single pose through the pose autoencoder
// in-place (normalize -> encode -> decode -> denormalize).
// Used to visually evaluate reconstruction quality.
static inline void NetworkApplyPoseAE(
    NetworkState* networkState,
    const AnimDatabase* db,
    std::span<float> pose)
{
    if (!networkState->poseAutoEncoder) return;

    const int poseDim = db->poseGenFeaturesComputeDim;
    if ((int)pose.size() != poseDim) return;

    // normalize: (raw - mean) / std * weight
    torch::Tensor normalized = torch::empty({ 1, poseDim });
    float* nPtr = normalized.data_ptr<float>();
    for (int d = 0; d < poseDim; ++d)
    {
        nPtr[d] = (pose[d] - db->poseGenFeaturesMean[d])
            / db->poseGenFeaturesStd[d]
            * db->poseGenFeaturesWeight[d];
    }

    // forward pass (eval mode)
    normalized = normalized.to(networkState->device);
    networkState->poseAutoEncoder->eval();
    torch::Tensor reconstructed =
        networkState->poseAutoEncoder->forward(normalized);
    reconstructed = reconstructed.to(torch::kCPU);

    // denormalize: raw = output / weight * std + mean
    const float* rPtr = reconstructed.data_ptr<float>();
    for (int d = 0; d < poseDim; ++d)
    {
        const float w = db->poseGenFeaturesWeight[d];
        const float denorm = (w > 1e-10f)
            ? (rPtr[d] / w * db->poseGenFeaturesStd[d]
                + db->poseGenFeaturesMean[d])
            : db->poseGenFeaturesMean[d];
        pose[d] = denorm;
    }
}