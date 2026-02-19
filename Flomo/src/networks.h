#pragma once

#include "definitions.h"
#include <string>
#include <filesystem>
#include <unordered_map>
#include <chrono>

extern "C" void cuda_init_context();

constexpr double AUTOSAVE_INTERVAL_SECONDS = 300.0;
constexpr double LOSS_LOG_INTERVAL_SECONDS = 5.0;

constexpr float BUDGET_WEIGHT_GLIMPSE_FLOW = 1.0f;
constexpr float BUDGET_WEIGHT_GLIMPSE_DECOMPRESSOR = 1.0f;
constexpr float GLIMPSE_DECOMPRESSOR_HEADSTART = 10.0f;
constexpr int GLIMPSE_FLOW_INFERENCE_STEPS = 8;

// switch between ReLU and GELU for all networks
#define USE_GELU
#ifdef USE_GELU
    #define NN_ACTIVATION() torch::nn::GELU()
    #define nn_activate(x) torch::gelu(x)
#else
    #define NN_ACTIVATION() torch::nn::RELU()
    #define nn_activate(x) torch::relu(x)
#endif

constexpr float LOSS_SMOOTHING_FACTOR = 0.1f;  // lerp factor: new loss weight (0=fully smoothed, 1=no smoothing)

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
// glimpse flow: 256 -> 512 -> 256, condition re-injected
//---------------------------------------------------------

inline GlimpseFlowModelImpl::GlimpseFlowModelImpl(int featureDim, int posePcaDim)
    : layer1(register_module("layer1",
          torch::nn::Linear(posePcaDim + featureDim + 1, 256))),
      layer2(register_module("layer2",
          torch::nn::Linear(256 + featureDim + 1, 512))),
      layer3(register_module("layer3",
          torch::nn::Linear(512 + featureDim + 1, 256))),
      outputLayer(register_module("outputLayer",
          torch::nn::Linear(256 + featureDim + 1, posePcaDim))),
      condTimeDim(featureDim + 1)
{
}

inline torch::Tensor GlimpseFlowModelImpl::forward(
    const torch::Tensor& xt,
    const torch::Tensor& condTime)
{
    torch::Tensor h = nn_activate(
        layer1->forward(torch::cat({xt, condTime}, 1)));
    h = nn_activate(
        layer2->forward(torch::cat({h, condTime}, 1)));
    h = nn_activate(
        layer3->forward(torch::cat({h, condTime}, 1)));
    return outputLayer->forward(torch::cat({h, condTime}, 1));
}

//---------------------------------------------------------
// glimpse decompressor: features re-injected, consistent 512 width
//---------------------------------------------------------

inline GlimpseDecompressorModelImpl::GlimpseDecompressorModelImpl(
    int featureDim, int pgDim, int segmentFlatDim)
    : layer1(register_module("layer1",
          torch::nn::Linear(pgDim + featureDim, 512))),
      layer2(register_module("layer2",
          torch::nn::Linear(512 + featureDim + pgDim, 512))),
      layer3(register_module("layer3",
          torch::nn::Linear(512 + featureDim + pgDim, 512))),
      outputLayer(register_module("outputLayer",
          torch::nn::Linear(512 + featureDim + pgDim, segmentFlatDim))),
      condDim(featureDim),
      poseDim(pgDim)
{
}

inline torch::Tensor GlimpseDecompressorModelImpl::forward(
    const torch::Tensor& futurePose,
    const torch::Tensor& cond)
{
    torch::Tensor h = nn_activate(
        layer1->forward(torch::cat({futurePose, cond}, 1)));
    h = nn_activate(
        layer2->forward(torch::cat({h, cond, futurePose}, 1)));
    h = nn_activate(
        layer3->forward(torch::cat({h, cond, futurePose}, 1)));
    return outputLayer->forward(torch::cat({h, cond, futurePose}, 1));
}

//---------------------------------------------------------
// glimpse flow: noise -> single future poseFeature
//---------------------------------------------------------

// compute per-dim std of the raw glimpse target vector for noise scaling in the flow.
// uses precomputed raw toe data.
static inline void ComputeGlimpseTargetStd(
    NetworkState* state,
    const AnimDatabase* db)
{
    if (db->precompRawToe.empty()) return;

    constexpr int glimpseDim = GLIMPSE_TOE_RAW_DIM;
    const int numSamples = std::min(10000, (int)db->legalStartFrames.size());
    if (numSamples <= 0) return;

    state->glimpseTargetStd.assign(glimpseDim, 0.0f);

    for (int i = 0; i < numSamples; ++i)
    {
        const int frame = SampleLegalSegmentStartFrame(db);
        std::span<const float> raw = db->precompRawToe.row_view(frame);

        for (int d = 0; d < glimpseDim; ++d)
        {
            state->glimpseTargetStd[d] += raw[d] * raw[d];
        }
    }

    for (int d = 0; d < glimpseDim; ++d)
    {
        state->glimpseTargetStd[d] = sqrtf(state->glimpseTargetStd[d] / (float)numSamples);
        if (state->glimpseTargetStd[d] < 1e-6f)
        {
            state->glimpseTargetStd[d] = 1e-6f;
        }
    }

    TraceLog(LOG_INFO, "Computed glimpse target std (%d raw dims, %d samples)",
        glimpseDim, numSamples);
}

static inline void NetworkInitGlimpseFlow(
    NetworkState* state,
    int featureDim,
    int posePcaK)
{
    if (featureDim <= 0 || posePcaK <= 0) return;

    NetworkEnsureDevice(state);

    state->glimpseFlowCondDim = featureDim;
    state->glimpseFlowGlimpseDim = posePcaK;

    state->glimpseFlow = GlimpseFlowModel(featureDim, posePcaK);
    state->glimpseFlow->to(state->device);

    state->glimpseFlowOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->glimpseFlow->parameters(),
            torch::optim::AdamOptions(1e-3).weight_decay(1e-4));

    state->glimpseFlowLoss = 0.0f;
    state->glimpseFlowIterations = 0;

    TraceLog(LOG_INFO,
        "GlimpseFlow initialized: cond=%d posePcaDim=%d",
        featureDim, posePcaK);
}

static inline void NetworkSaveGlimpseFlow(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->glimpseFlow) return;
    const std::string path = folderPath + "/glimpseFlow.bin";
    try
    {
        torch::save(state->glimpseFlow, path);
        TraceLog(LOG_INFO, "Saved GlimpseFlow to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save GlimpseFlow: %s", e.what());
    }
}

static inline void NetworkLoadGlimpseFlow(
    NetworkState* state,
    int featureDim,
    int posePcaK,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/glimpseFlow.bin";
    if (!std::filesystem::exists(path)) return;
    try
    {
        NetworkInitGlimpseFlow(state, featureDim, posePcaK);
        torch::load(state->glimpseFlow, path);
        state->glimpseFlow->to(state->device);
        TraceLog(LOG_INFO, "Loaded GlimpseFlow from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load GlimpseFlow: %s", e.what());
    }
}

//---------------------------------------------------------
// glimpse decompressor: (features, future pose) -> segment
//---------------------------------------------------------

static inline void NetworkInitGlimpseDecompressor(
    NetworkState* state,
    int featureDim,
    int posePcaK,
    int segmentPcaK)
{
    if (featureDim <= 0 || posePcaK <= 0 || segmentPcaK <= 0) return;

    NetworkEnsureDevice(state);

    state->glimpseDecompCondDim = featureDim;
    state->glimpseDecompGlimpseDim = posePcaK;
    state->glimpseDecompSegK = segmentPcaK;

    state->glimpseDecompressor = GlimpseDecompressorModel(featureDim, posePcaK, segmentPcaK);
    state->glimpseDecompressor->to(state->device);

    state->glimpseDecompressorOptimizer =
        std::make_shared<torch::optim::Adam>(
            state->glimpseDecompressor->parameters(),
            torch::optim::AdamOptions(1e-3).weight_decay(1e-4));

    state->glimpseDecompressorLoss = 0.0f;
    state->glimpseDecompressorIterations = 0;

    TraceLog(LOG_INFO,
        "GlimpseDecompressor initialized: toe=%d cond=%d -> 512 -> 512 -> 512 -> %d (cond re-injected)",
        posePcaK, featureDim, segmentPcaK);
}

static inline void NetworkSaveGlimpseDecompressor(
    const NetworkState* state,
    const std::string& folderPath)
{
    if (!state->glimpseDecompressor) return;
    const std::string path = folderPath + "/glimpseDecompressor.bin";
    try
    {
        torch::save(state->glimpseDecompressor, path);
        TraceLog(LOG_INFO, "Saved GlimpseDecompressor to: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to save GlimpseDecompressor: %s", e.what());
    }
}

static inline void NetworkLoadGlimpseDecompressor(
    NetworkState* state,
    int featureDim,
    int posePcaK,
    int segmentPcaK,
    const std::string& folderPath)
{
    const std::string path = folderPath + "/glimpseDecompressor.bin";
    if (!std::filesystem::exists(path)) return;
    try
    {
        NetworkInitGlimpseDecompressor(state, featureDim, posePcaK, segmentPcaK);
        torch::load(state->glimpseDecompressor, path);
        state->glimpseDecompressor->to(state->device);
        TraceLog(LOG_INFO, "Loaded GlimpseDecompressor from: %s", path.c_str());
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "Failed to load GlimpseDecompressor: %s", e.what());
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

    fprintf(f, "time,glimpseFlow,glimpseDecomp\n");
    const int n = (int)state->lossHistoryTime.size();
    for (int i = 0; i < n; ++i)
    {
        fprintf(f, "%.1f,%.8f,%.8f\n",
            state->lossHistoryTime[i],
            i < (int)state->glimpseFlowLossHistory.size() ? state->glimpseFlowLossHistory[i] : 0.0f,
            i < (int)state->glimpseDecompressorLossHistory.size() ? state->glimpseDecompressorLossHistory[i] : 0.0f);
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
    state->glimpseFlowLossHistory.clear();
    state->glimpseDecompressorLossHistory.clear();

    // skip header
    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return; }

    while (fgets(line, sizeof(line), f))
    {
        float t = 0, glimpseF = 0, glimpseD = 0;
        const int cols = sscanf(line, "%f,%f,%f", &t, &glimpseF, &glimpseD);
        if (cols < 1) break;
        state->lossHistoryTime.push_back(t);
        state->glimpseFlowLossHistory.push_back(cols >= 2 ? glimpseF : 0.0f);
        state->glimpseDecompressorLossHistory.push_back(cols >= 3 ? glimpseD : 0.0f);
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
    const AnimDatabase* db,
    const std::string& folderPath)
{
    AnimDatabaseSaveDerived(db, folderPath);
    NetworkSaveGlimpseFlow(state, folderPath);
    NetworkSaveGlimpseDecompressor(state, folderPath);
    LossHistorySave(state, folderPath);
}

static inline void NetworkLoadAll(
    NetworkState* state,
    AnimDatabase* db,
    const std::string& folderPath)
{
    LOG_PROFILE_SCOPE(NetworkLoadAll);

    // load clusters + PCA from file, or compute if missing
    if (!AnimDatabaseLoadDerived(db, folderPath))
    {
        if (db->clusterCount == 0)
            AnimDatabaseClusterFeatures2(db);
        AnimDatabaseComputeSegmentPCA(db);
    }
    AnimDatabaseComputeFeaturePCA(db);
    AnimDatabasePrecomputeTrainingData(db);

    if (db->pcaSegmentK > 0 && db->pcaFeatureK > 0)
    {
        const int glimpseDim = GLIMPSE_TOE_RAW_DIM;
        const int condDim = db->pcaFeatureK;
        NetworkLoadGlimpseFlow(state, condDim, glimpseDim, folderPath);
        NetworkLoadGlimpseDecompressor(
            state, condDim, glimpseDim, db->pcaSegmentK, folderPath);
    }
    LossHistoryLoad(state, folderPath);
}


//---------------------------------------------------------
// glimpse flow training: noise -> single future poseFeature
//---------------------------------------------------------

// GlimpseFlow training: learn to map noise -> future toe positions/velocities.
//
// the big picture:
//   "given what the character is doing right now, where will the feet be in 0.1s and 0.3s?"
//
// the flow operates directly on 16 raw toe dims (GlimpseFeatures):
//   2 future times x 2 toes x (posX, posZ, velX, velZ) in current root space.
//   no PCA — the flow outputs raw positions and velocities.
//
// the conditioning features go through their own normalization pipeline:
//   raw features -> (x - mean) / typeStd * typeWeight -> PCA to 20 dims
// these describe "what is the character doing" (foot positions, velocities, trajectory).
// the decompressor then takes (raw toe dims + condition) and produces a motion segment.
//
static inline void NetworkTrainGlimpseFlowForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->glimpseFlow) return;
    if (budgetSeconds <= 0.0) return;
    if (db->precompFeaturePCA.empty() || db->precompRawToe.empty()) return;
    if (db->pcaFeatureK <= 0) return;

    const int batchSize = 256;
    const int condDim = db->pcaFeatureK;
    constexpr int glimpseDim = GLIMPSE_TOE_RAW_DIM;
    const Clock::time_point start = Clock::now();

    state->glimpseFlow->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor condBatch = torch::empty({batchSize, condDim});
            torch::Tensor x1Batch = torch::empty({batchSize, glimpseDim});
            float* cPtr = condBatch.data_ptr<float>();
            float* xPtr = x1Batch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalSegmentStartFrame(db);
                memcpy(cPtr + b * condDim, db->precompFeaturePCA.row_view(frame).data(),
                    condDim * sizeof(float));
                memcpy(xPtr + b * glimpseDim, db->precompRawToe.row_view(frame).data(),
                    glimpseDim * sizeof(float));
            }

            condBatch = condBatch.to(state->device);
            x1Batch = x1Batch.to(state->device);

            // flow matching setup: random time t in [0,1], and a starting noise sample x0.
            // at t=0 we're at pure noise, at t=1 we should be at the real data.
            torch::Tensor t = torch::rand(
                {batchSize, 1},
                torch::TensorOptions().device(state->device));
            torch::Tensor x0 = torch::randn(
                {batchSize, glimpseDim},
                torch::TensorOptions().device(state->device));

            // scale the noise to match the per-dim std of the targets
            if (!state->glimpseTargetStd.empty())
            {
                torch::Tensor stdTensor = torch::from_blob(
                    (void*)state->glimpseTargetStd.data(),
                    {1, glimpseDim}, torch::kFloat32)
                    .clone().to(state->device);
                x0 = x0 * stdTensor;
            }

            // linear interpolation between noise (t=0) and target (t=1)
            torch::Tensor xt = (1.0f - t) * x0 + t * x1Batch;

            // the network predicts the velocity field (x1 - x0) that takes us from noise to target
            torch::Tensor target = x1Batch - x0;

            // condition = [normalized MM features, time] concatenated
            torch::Tensor condTime = torch::cat({condBatch, t}, 1);

            state->glimpseFlowOptimizer->zero_grad();
            torch::Tensor predicted = state->glimpseFlow->forward(xt, condTime);
            torch::Tensor loss = torch::mse_loss(predicted, target);

            loss.backward();
            state->glimpseFlowOptimizer->step();

            state->glimpseFlowLoss = loss.item<float>();
            state->glimpseFlowLossSmoothed = state->glimpseFlowIterations == 0
                ? state->glimpseFlowLoss
                : state->glimpseFlowLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                  + state->glimpseFlowLoss * LOSS_SMOOTHING_FACTOR;
            state->glimpseFlowIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "GlimpseFlow Training Error: %s", e.what());
        state->isTraining = false;
    }
}

//---------------------------------------------------------
// glimpse decompressor training: (features, future pose) -> segment
//---------------------------------------------------------

// GlimpseDecompressor training: given future toe info and current features, predict the
// full segment of animation that connects "now" to "there".
//
// two-stage glimpse pipeline:
//   1. GlimpseFlow:         "where will the toes go?"  -> 16 raw toe dims
//   2. GlimpseDecompressor: "ok, fill in the motion to get there" -> 250 PCA segment coefficients
//
// the decompressor sees two things:
//   - glimpseVec (16 dims): raw toe positions/velocities (from flow or ground truth)
//   - condition  (20 dims): where we are now (MM features PCA)
// and produces:
//   - segment PCA coefficients (250 dims): the full motion to get there
//
// during training we use ground truth raw toe data (not the flow's output) so the
// decompressor learns from clean targets.
//
static inline void NetworkTrainGlimpseDecompressorForTime(
    NetworkState* state,
    const AnimDatabase* db,
    double budgetSeconds)
{
    if (!state->glimpseDecompressor) return;
    if (budgetSeconds <= 0.0) return;
    if (db->precompSegmentPCA.empty() || db->precompFeaturePCA.empty()
        || db->precompRawToe.empty()) return;
    if (db->pcaSegmentK <= 0 || db->pcaFeatureK <= 0) return;

    const int batchSize = 256;
    const int condDim = db->pcaFeatureK;
    constexpr int glimpseDim = GLIMPSE_TOE_RAW_DIM;
    const int segK = db->pcaSegmentK;
    const Clock::time_point start = Clock::now();

    state->glimpseDecompressor->train();

    try
    {
        while (ElapsedSeconds(start) < budgetSeconds)
        {
            torch::Tensor condBatch = torch::empty({batchSize, condDim});
            torch::Tensor toeBatch = torch::empty({batchSize, glimpseDim});
            torch::Tensor targetBatch = torch::empty({batchSize, segK});
            float* cPtr = condBatch.data_ptr<float>();
            float* pPtr = toeBatch.data_ptr<float>();
            float* tPtr = targetBatch.data_ptr<float>();

            for (int b = 0; b < batchSize; ++b)
            {
                const int frame = SampleLegalSegmentStartFrame(db);
                memcpy(cPtr + b * condDim, db->precompFeaturePCA.row_view(frame).data(),
                    condDim * sizeof(float));
                memcpy(pPtr + b * glimpseDim, db->precompRawToe.row_view(frame).data(),
                    glimpseDim * sizeof(float));
                memcpy(tPtr + b * segK, db->precompSegmentPCA.row_view(frame).data(),
                    segK * sizeof(float));
            }

            condBatch = condBatch.to(state->device);
            toeBatch = toeBatch.to(state->device);
            targetBatch = targetBatch.to(state->device);

            state->glimpseDecompressorOptimizer->zero_grad();
            torch::Tensor predicted = state->glimpseDecompressor->forward(toeBatch, condBatch);
            torch::Tensor loss = torch::mse_loss(predicted, targetBatch);

            loss.backward();
            state->glimpseDecompressorOptimizer->step();

            state->glimpseDecompressorLoss = loss.item<float>();
            state->glimpseDecompressorLossSmoothed = state->glimpseDecompressorIterations == 0
                ? state->glimpseDecompressorLoss
                : state->glimpseDecompressorLossSmoothed * (1.0f - LOSS_SMOOTHING_FACTOR)
                  + state->glimpseDecompressorLoss * LOSS_SMOOTHING_FACTOR;
            state->glimpseDecompressorIterations++;
        }
    }
    catch (const std::exception& e)
    {
        TraceLog(LOG_ERROR, "GlimpseDecompressor Training Error: %s", e.what());
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
    if (db->legalStartFrames.empty()) return 0.0;

    const Clock::time_point wallStart = Clock::now();

    // auto-init glimpse flow immediately (operates on raw toe dims, no PCA)
    if (!state->glimpseFlow && BUDGET_WEIGHT_GLIMPSE_FLOW > 0.0f
        && db->pcaFeatureK > 0)
    {
        const int condDim = db->pcaFeatureK;
        NetworkInitGlimpseFlow(state, condDim, GLIMPSE_TOE_RAW_DIM);
        ComputeGlimpseTargetStd(state, db);
    }
    // decompressor waits for flow headstart so it never trains on pure noise
    if (!state->glimpseDecompressor && BUDGET_WEIGHT_GLIMPSE_DECOMPRESSOR > 0.0f
        && db->pcaFeatureK > 0 && db->pcaSegmentK > 0
        && state->glimpseFlow
        && state->trainingElapsedSeconds >= GLIMPSE_DECOMPRESSOR_HEADSTART)
    {
        const int condDim = db->pcaFeatureK;
        NetworkInitGlimpseDecompressor(
            state, condDim, GLIMPSE_TOE_RAW_DIM,
            db->pcaSegmentK);
    }

    const bool hasGlimpseFlow = static_cast<bool>(state->glimpseFlow);
    const bool hasGlimpseDecomp = static_cast<bool>(state->glimpseDecompressor);

    const float wGlimpseFlow = hasGlimpseFlow ? BUDGET_WEIGHT_GLIMPSE_FLOW : 0.0f;
    const float wGlimpseDecomp = hasGlimpseDecomp ? BUDGET_WEIGHT_GLIMPSE_DECOMPRESSOR : 0.0f;
    const float totalWeight = wGlimpseFlow + wGlimpseDecomp;

    if (totalWeight <= 0.0f) return 0.0;

    const double glimpseFlowBudget = totalBudgetSeconds * wGlimpseFlow / totalWeight;
    const double glimpseDecompBudget = totalBudgetSeconds * wGlimpseDecomp / totalWeight;

    if (state->glimpseFlow && glimpseFlowBudget > 0.0)
        NetworkTrainGlimpseFlowForTime(state, db, glimpseFlowBudget);
    if (state->glimpseDecompressor && glimpseDecompBudget > 0.0)
        NetworkTrainGlimpseDecompressorForTime(state, db, glimpseDecompBudget);

    const double wallElapsed = ElapsedSeconds(wallStart);
    state->trainingElapsedSeconds += wallElapsed;
    state->timeSinceLastAutoSave += wallElapsed;
    state->timeSinceLastLossLog += wallElapsed;

    if (state->timeSinceLastLossLog >= LOSS_LOG_INTERVAL_SECONDS)
    {
        state->timeSinceLastLossLog = 0.0;
        state->lossHistoryTime.push_back((float)state->trainingElapsedSeconds);
        state->glimpseFlowLossHistory.push_back(state->glimpseFlowIterations >= 100 ? state->glimpseFlowLossSmoothed : 0.0f);
        state->glimpseDecompressorLossHistory.push_back(state->glimpseDecompressorIterations >= 100 ? state->glimpseDecompressorLossSmoothed : 0.0f);
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

    LOG_PROFILE_SCOPE(NetworkInitAllForTraining);

    // cluster features for stratified training sampling (lazy, only once)
    if (db->clusterCount == 0)
        AnimDatabaseClusterFeatures2(db);

    // PCA on segments and features (lazy, only once). toe PCA no longer needed.
    AnimDatabaseComputeSegmentPCA(db);
    AnimDatabaseComputeFeaturePCA(db);

    // precompute all per-frame training data (segment PCA, feature PCA, raw toe)
    AnimDatabasePrecomputeTrainingData(db);

    // glimpse networks auto-init inside NetworkTrainAll

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;
    state->isTraining = true;

    TraceLog(LOG_INFO, "Training initialized.");
}

// reset all networks to null (e.g. after database rebuild
// which invalidates feature dimensions)
static inline void NetworkResetAll(NetworkState* state)
{
    state->isTraining = false;

    state->glimpseFlow = nullptr;
    state->glimpseFlowOptimizer = nullptr;
    state->glimpseFlowLoss = 0.0f;
    state->glimpseFlowLossSmoothed = 0.0f;
    state->glimpseFlowIterations = 0;
    state->glimpseTargetStd.clear();

    state->glimpseDecompressor = nullptr;
    state->glimpseDecompressorOptimizer = nullptr;
    state->glimpseDecompressorLoss = 0.0f;
    state->glimpseDecompressorLossSmoothed = 0.0f;
    state->glimpseDecompressorIterations = 0;

    state->lossHistoryTime.clear();
    state->glimpseFlowLossHistory.clear();
    state->glimpseDecompressorLossHistory.clear();
    state->timeSinceLastLossLog = 0.0;

    state->trainingElapsedSeconds = 0.0;
    state->timeSinceLastAutoSave = 0.0;

    state->glimpseFlowCondDim = 0;
    state->glimpseFlowGlimpseDim = 0;
    state->glimpseDecompCondDim = 0;
    state->glimpseDecompGlimpseDim = 0;
    state->glimpseDecompSegK = 0;

    TraceLog(LOG_INFO, "All networks reset.");
}

//---------------------------------------------------------
// glimpse combined inference: flow -> future pose -> decompressor -> segment
//---------------------------------------------------------

// Glimpse inference: the full pipeline at runtime.
//
// this is the payoff of all that training. we take the character's current state
// (raw MM features from the live query) and produce a complete animation segment
// for the next 0.3 seconds. here's the journey through all the spaces:
//
//   INPUT: raw MM features (foot positions, velocities, future trajectory, etc.)
//     these come straight from the live character state, in physical units.
//
//   step 0: normalize the raw features
//     raw -> (clamped - mean) / typeStd * typeWeight
//     this puts them in the same space the networks were trained on.
//     we clamp to data bounds here (unlike training which uses unclamped augmented features)
//     because at runtime we want to be conservative with out-of-distribution inputs.
//
//   step 1: GlimpseFlow — sample glimpse vector (K PCA + 6 motion dims)
//     start from shaped noise (gaussian scaled by per-component std, halved for less
//     randomness) and iteratively push it through the learned velocity field.
//     after all steps, x contains K PCA pose coefficients + 6 displacement/yaw values.
//
//   step 1b (optional): reconstruct the last pose (0.3s) for visualization
//     PCA coefficients -> normalized pose (~150) -> denormalized raw pose (~150)
//
//   step 2: GlimpseDecompressor — expand future poses into a full segment
//     (all PCA pose coefficients, normalized features) -> 128 PCA segment coefficients
//     the decompressor sees both future poses and the current features.
//
//   step 3: reconstruct the segment from PCA
//     128 PCA coefficients -> flat normalized segment -> denormalized
//     then reshape into [segFrames x pgDim] for the caller to use.
//
static inline bool NetworkPredictGlimpse(
    NetworkState* state,
    const AnimDatabase* db,
    const std::vector<float>& rawQuery,
    Array2D<float>& /*out*/ segment,
    GlimpseFeatures* /*out*/ outGlimpse)
{
    if (!state->glimpseFlow) return false;
    if (!state->glimpseDecompressor) return false;
    if ((int)rawQuery.size() != db->featureDim) return false;
    if (db->pcaFeatureK <= 0) return false;

    const int featureDim = db->featureDim;
    const int condDim = db->pcaFeatureK;
    const int pgDim = db->poseGenFeaturesComputeDim;
    constexpr int glimpseDim = GLIMPSE_TOE_RAW_DIM;
    const int segFrames = db->poseGenSegmentFrameCount;

    segment.resize(segFrames, pgDim);

    // step 0: normalize the live MM query, then project to PCA feature space
    std::vector<float> normQuery(featureDim);
    NormalizeFeatureQuery(db, rawQuery.data(), normQuery.data());
    torch::Tensor condTensor = torch::empty({1, condDim});
    PcaProjectFeature(db, normQuery.data(), condTensor.data_ptr<float>());

    state->glimpseFlow->eval();
    state->glimpseDecompressor->eval();
    condTensor = condTensor.to(state->device);

    torch::Tensor result;
    {
        torch::NoGradGuard noGrad;

        // step 1: flow — walk from noise to 16 raw toe dims.
        // we start at half the distribution-matched noise.
        torch::Tensor x = torch::randn(
            {1, glimpseDim}, torch::TensorOptions().device(state->device));
        if (!state->glimpseTargetStd.empty())
        {
            torch::Tensor stdTensor = torch::from_blob(
                (void*)state->glimpseTargetStd.data(),
                {1, glimpseDim}, torch::kFloat32)
                .clone().to(state->device);
            x = x * stdTensor * 0.5f;
        }
        else
        {
            x = x * 0.5f;
        }

        constexpr int steps = GLIMPSE_FLOW_INFERENCE_STEPS;

        for (int step = 0; step < steps; ++step)
        {
            const float t = (float)step / steps;
            torch::Tensor tTensor = torch::full(
                {1, 1}, t,
                torch::TensorOptions().device(state->device));
            torch::Tensor condTime = torch::cat({condTensor, tTensor}, 1);

            torch::Tensor v = state->glimpseFlow->forward(x, condTime);
            x = x + v * (1.0f / steps);
        }

        // x is now [1, glimpseDim]: 16 raw toe dims

        // deserialize raw output directly into GlimpseFeatures
        if (outGlimpse != nullptr)
        {
            torch::Tensor xCpu = x.to(torch::kCPU);
            outGlimpse->DeserializeFrom(xCpu.data_ptr<float>());
        }

        // step 2: decompressor takes raw toe dims + features
        // and produces segment PCA coefficients
        result = state->glimpseDecompressor->forward(x, condTensor);
        result = result.to(torch::kCPU);
    }

    // step 3: reconstruct the full segment from the PCA coefficients,
    // then denormalize each frame back to raw pose feature space
    const int flatDim = db->poseGenSegmentFlatDim;
    std::vector<float> normalizedFlat(flatDim);
    PcaReconstructSegment(db, result.data_ptr<float>(), normalizedFlat.data());

    for (int f = 0; f < segFrames; ++f)
    {
        std::span<float> dst = segment.row_view(f);
        for (int d = 0; d < pgDim; ++d)
        {
            const float w = db->poseGenFeaturesWeight[d];
            const float denorm = (w > 1e-10f)
                ? (normalizedFlat[f * pgDim + d] / w
                    * db->poseGenFeaturesStd[d]
                    + db->poseGenFeaturesMean[d])
                : db->poseGenFeaturesMean[d];
            dst[d] = denorm;
        }
    }

    return true;
}

//---------------------------------------------------------
// training thread: stage weights and main loop
//---------------------------------------------------------

constexpr double TRAINING_THREAD_BUDGET_SECONDS = 0.6;
constexpr double WEIGHT_STAGING_INTERVAL_SECONDS = 10.0;

// deep-copy model parameters from training state to staging area (CPU).
// called on the training thread; locks the mutex briefly.
static void StageWeightsFromTraining(
    TrainingThreadControl* ctrl,
    const NetworkState* trainingState)
{
    GlimpseFlowModel flowClone = nullptr;
    GlimpseDecompressorModel decompClone = nullptr;

    {
        torch::NoGradGuard noGrad;

        if (trainingState->glimpseFlow)
        {
            flowClone = GlimpseFlowModel(
                trainingState->glimpseFlowCondDim,
                trainingState->glimpseFlowGlimpseDim);
            flowClone->to(torch::kCPU);

            std::vector<torch::Tensor> srcParams = trainingState->glimpseFlow->parameters();
            std::vector<torch::Tensor> dstParams = flowClone->parameters();
            assert(srcParams.size() == dstParams.size());
            for (size_t i = 0; i < srcParams.size(); i++)
            {
                dstParams[i].copy_(srcParams[i].cpu());
            }
            flowClone->eval();
        }

        if (trainingState->glimpseDecompressor)
        {
            decompClone = GlimpseDecompressorModel(
                trainingState->glimpseDecompCondDim,
                trainingState->glimpseDecompGlimpseDim,
                trainingState->glimpseDecompSegK);
            decompClone->to(torch::kCPU);

            std::vector<torch::Tensor> srcParams = trainingState->glimpseDecompressor->parameters();
            std::vector<torch::Tensor> dstParams = decompClone->parameters();
            assert(srcParams.size() == dstParams.size());
            for (size_t i = 0; i < srcParams.size(); i++)
            {
                dstParams[i].copy_(srcParams[i].cpu());
            }
            decompClone->eval();
        }
    }

    // lock briefly to swap staged models
    {
        std::lock_guard<std::mutex> lock(ctrl->stagingMutex);
        ctrl->stagedGlimpseFlow = std::move(flowClone);
        ctrl->stagedGlimpseDecompressor = std::move(decompClone);
        ctrl->stagedGlimpseTargetStd = trainingState->glimpseTargetStd;
        ctrl->stagingReady = true;

        ctrl->stagedLossHistoryTime = trainingState->lossHistoryTime;
        ctrl->stagedGlimpseFlowLossHistory = trainingState->glimpseFlowLossHistory;
        ctrl->stagedGlimpseDecompressorLossHistory = trainingState->glimpseDecompressorLossHistory;
        ctrl->lossHistoryDirty = true;
    }
}

// training thread entry point. Runs until stopRequested is set.
static void TrainingThreadMain(
    NetworkState* state,
    const AnimDatabase* db,
    TrainingThreadControl* ctrl)
{
    ctrl->isRunning.store(true);
    double timeSinceLastStage = 0.0;
    double timeSinceLastAutoSave = 0.0;

    TraceLog(LOG_INFO, "Training thread started.");

    while (!ctrl->stopRequested.load())
    {
        const Clock::time_point roundStart = Clock::now();

        // train both networks for ~0.6s total (~0.3s each)
        NetworkTrainAll(state, db, TRAINING_THREAD_BUDGET_SECONDS);

        const double roundElapsed = ElapsedSeconds(roundStart);
        timeSinceLastStage += roundElapsed;
        timeSinceLastAutoSave += roundElapsed;

        // publish status via atomics (lock-free, main thread can read anytime)
        ctrl->glimpseFlowLossSmoothed.store(state->glimpseFlowLossSmoothed);
        ctrl->glimpseFlowIterations.store(state->glimpseFlowIterations);
        ctrl->glimpseDecompressorLossSmoothed.store(state->glimpseDecompressorLossSmoothed);
        ctrl->glimpseDecompressorIterations.store(state->glimpseDecompressorIterations);
        ctrl->trainingElapsedSeconds.store(state->trainingElapsedSeconds);

        // stage weights every ~10 seconds
        if (timeSinceLastStage >= WEIGHT_STAGING_INTERVAL_SECONDS)
        {
            timeSinceLastStage = 0.0;
            StageWeightsFromTraining(ctrl, state);
            TraceLog(LOG_INFO, "Training thread: staged weights (%.0fs elapsed)",
                state->trainingElapsedSeconds);
        }

        // autosave every 5 minutes
        if (timeSinceLastAutoSave >= AUTOSAVE_INTERVAL_SECONDS)
        {
            timeSinceLastAutoSave = 0.0;
            const std::string autoSavePath = "saved/autosave";
            NetworkSaveAll(state, db, autoSavePath);
            TraceLog(LOG_INFO, "Training thread: autosaved to %s", autoSavePath.c_str());
        }
    }

    // final stage so main thread gets the latest weights
    StageWeightsFromTraining(ctrl, state);

    ctrl->isRunning.store(false);
    TraceLog(LOG_INFO, "Training thread stopped.");
}

