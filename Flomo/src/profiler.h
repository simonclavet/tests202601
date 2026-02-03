#pragma once

//----------------------------------------------------------------------------------
// Profiling
//----------------------------------------------------------------------------------


// Profiling only available on Windows
#if defined(ENABLE_PROFILE) && defined(_WIN32)

#include <stdint.h>

// Windows headers must come before raylib to avoid conflicts
#define WIN32_LEAN_AND_MEAN
#define NOGDI             // Excludes GDI (avoids Rectangle conflict)
#define NOUSER            // Excludes USER (avoids CloseWindow, ShowCursor, DrawText conflicts)
#include <windows.h>

enum
{
    // Max number of profile records (profiled code locations)
    PROFILE_RECORD_MAX = 512,

    // Maximum number of timer samples per record
    PROFILE_RECORD_SAMPLE_MAX = 128,
};

// A single record for a profiled code location with a cyclic buffer of start and end times.
typedef struct
{
    const char* name;
    uint32_t idx;
    uint32_t num;

    struct {
        LARGE_INTEGER start;
        LARGE_INTEGER end;
    } samples[PROFILE_RECORD_SAMPLE_MAX];

} ProfileRecord;

// Structure containing space for all profiled code locations
typedef struct
{
    uint32_t num;
    LARGE_INTEGER freq;
    ProfileRecord* records[PROFILE_RECORD_MAX];

} ProfileRecordData;

// Global variable storing all the profile record data
static ProfileRecordData globalProfileRecords;

// Init the Profile Record Data. Must be called at program start
static void ProfileRecordDataInit()
{
    globalProfileRecords.num = 0;
    QueryPerformanceFrequency(&globalProfileRecords.freq);
    memset(globalProfileRecords.records, 0, sizeof(ProfileRecord*) * PROFILE_RECORD_MAX);
}

// If uninitialized, then initialize the profile record, then store the start time
static inline void ProfileRecordBegin(ProfileRecord* record, const char* name)
{
    if (!record->name && globalProfileRecords.num < PROFILE_RECORD_MAX)
    {
        record->name = name;
        record->idx = 0;
        record->num = 0;
        globalProfileRecords.records[globalProfileRecords.num] = record;
        globalProfileRecords.num++;
    }

    QueryPerformanceCounter(&record->samples[record->idx].start);
}

// Store the end time and increment the record sample num
static inline void ProfileRecordEnd(ProfileRecord* record)
{
    QueryPerformanceCounter(&record->samples[record->idx].end);
    record->idx = (record->idx + 1) % PROFILE_RECORD_SAMPLE_MAX;
    record->num++;
}

// Tickers record a rolling average of Profile Record durations in microseconds
typedef struct
{
    uint64_t unitScale;
    double alpha;
    uint32_t samples[PROFILE_RECORD_MAX];
    uint64_t iterations[PROFILE_RECORD_MAX];
    double averages[PROFILE_RECORD_MAX];
    double times[PROFILE_RECORD_MAX];

} ProfileTickers;

// Global profile tickers data
static ProfileTickers globalProfileTickers;

// Initialize ticker data
static inline void ProfileTickersInit()
{
    globalProfileTickers.unitScale = 1000000; // Microseconds
    globalProfileTickers.alpha = 0.9f;
    memset(globalProfileTickers.samples, 0, sizeof(uint32_t) * PROFILE_RECORD_MAX);
    memset(globalProfileTickers.iterations, 0, sizeof(uint64_t) * PROFILE_RECORD_MAX);
    memset(globalProfileTickers.averages, 0, sizeof(double) * PROFILE_RECORD_MAX);
    memset(globalProfileTickers.times, 0, sizeof(double) * PROFILE_RECORD_MAX);
}

// Update tickers and compute the rolling average of the duration
static inline void ProfileTickersUpdate()
{
    for (int i = 0; i < (int)globalProfileRecords.num; i++)
    {
        ProfileRecord* record = globalProfileRecords.records[i];

        if (record && record->name)
        {
            globalProfileTickers.samples[i] = record->num;

            int bufferedSampleNum = record->num < PROFILE_RECORD_SAMPLE_MAX ? record->num : PROFILE_RECORD_SAMPLE_MAX;

            for (int j = 0; j < bufferedSampleNum; j++)
            {
                double time = (double)((
                    record->samples[j].end.QuadPart -
                    record->samples[j].start.QuadPart) * globalProfileTickers.unitScale) /
                        (double)globalProfileRecords.freq.QuadPart;

                globalProfileTickers.iterations[i]++;
                globalProfileTickers.averages[i] = globalProfileTickers.alpha * globalProfileTickers.averages[i] + (1.0 - globalProfileTickers.alpha) * time;
                globalProfileTickers.times[i] = globalProfileTickers.averages[i] / (1.0 - pow(globalProfileTickers.alpha, (double)globalProfileTickers.iterations[i]));
            }

            // Flush Samples
            record->idx = 0;
            record->num = 0;
        }
    }
}

#define PROFILE_INIT() ProfileRecordDataInit();
#define PROFILE_BEGIN(NAME) static ProfileRecord __PROFILE_RECORD_##NAME; ProfileRecordBegin(&__PROFILE_RECORD_##NAME, #NAME);
#define PROFILE_END(NAME) ProfileRecordEnd(&__PROFILE_RECORD_##NAME);

#define PROFILE_TICKERS_INIT() ProfileTickersInit();
#define PROFILE_TICKERS_UPDATE() ProfileTickersUpdate()

#else
#define PROFILE_INIT()
#define PROFILE_BEGIN(NAME)
#define PROFILE_END(NAME)

#define PROFILE_TICKERS_INIT()
#define PROFILE_TICKERS_UPDATE()
#endif


//--------------------------------------
// One-off console timing (for loading/building operations): NOT IN THE TICK!
//--------------------------------------

//#ifdef ENABLE_PROFILE // do it all the time because they should not be done in the tick
#define LOG_PROFILE_START(name) \
        std::chrono::high_resolution_clock::time_point __logProfile_##name##_start = std::chrono::high_resolution_clock::now();

#define LOG_PROFILE_END(name) \
        do { \
            std::chrono::high_resolution_clock::time_point __logProfile_##name##_end = std::chrono::high_resolution_clock::now(); \
            std::chrono::milliseconds __logProfile_##name##_duration = std::chrono::duration_cast<std::chrono::milliseconds>( \
                __logProfile_##name##_end - __logProfile_##name##_start); \
            TraceLog(LOG_INFO, #name ": %lld ms", (long long)__logProfile_##name##_duration.count()); \
        } while(0)
//#else
//#define LOG_PROFILE_START(name)
//#define LOG_PROFILE_END(name)
//#endif