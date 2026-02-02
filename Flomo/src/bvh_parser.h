#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <string>
#include <vector>

#include "raylib.h"
#include "raymath.h"
#include "utils.h"

//----------------------------------------------------------------------------------
// BVH Parser - Text Parsing Utilities
//----------------------------------------------------------------------------------

enum
{
    BVH_PARSER_ERR_MAX = 512,
};

// Simple parser that keeps track of rows, and cols in a string and so can provide slightly
// nicer error messages. Has ability to peek at next character and advance the input
struct BVHParser
{
    std::string filename;   // now a C++ string
    int offset = 0;
    const char* data;
    int row_view = 0;
    int col = 0;
    char err[BVH_PARSER_ERR_MAX];
};

// Initialize the Parser
static inline void BVHParserInit(BVHParser* par, const char* filename, const char* data)
{
    par->filename = filename ? filename : "";
    par->offset = 0;
    par->data = data;
    par->row_view = 0;
    par->col = 0;
    par->err[0] = '\0';
}

// Peek at the next character in the stream
static inline char BVHParserPeek(const BVHParser* par)
{
    return par->data[par->offset];
}

// Peek forward N steps in the stream. Does not check the stream is long enough.
static inline char BVHParserPeekForward(const BVHParser* par, int steps)
{
    return par->data[par->offset + steps];
}

// Checks the current character matches the given input
static inline bool BVHParserMatch(const BVHParser* par, char match)
{
    return match == par->data[par->offset];
}

// Checks the current character matches one of the given characters
static inline bool BVHParserOneOf(const BVHParser* par, const char* matches)
{
    return strchr(matches, par->data[par->offset]);
}

// Checks the following characters in the stream match the prefix (in a caseless way)
static inline bool BVHParserStartsWithCaseless(const BVHParser* par, const char* prefix)
{
    const char* start = par->data + par->offset;
    while (*prefix)
    {
        if (tolower((unsigned char)*prefix) != tolower((unsigned char)*start)) { return false; }
        prefix++;
        start++;
    }

    return true;
}

// Advances the stream forward one
static inline void BVHParserInc(BVHParser* par)
{
    if (par->data[par->offset] == '\n')
    {
        par->row_view++;
        par->col = 0;
    }
    else
    {
        par->col++;
    }

    par->offset++;
}

// Advances the stream forward "num" characters
static inline void BVHParserAdvance(BVHParser* par, int num)
{
    for (int i = 0; i < num; i++) { BVHParserInc(par); }
}

// Gets the human readable name of a particular character
static inline const char* BVHParserCharName(char c)
{
    static char parserCharName[2];

    switch (c)
    {
    case '\0': return "end of file";
    case '\r': return "new line";
    case '\n': return "new line";
    case '\t': return "tab";
    case '\v': return "vertical tab";
    case '\b': return "backspace";
    case '\f': return "form feed";
    default:
        parserCharName[0] = c;
        parserCharName[1] = '\0';
        return parserCharName;
    }
}

// Prints a formatted error to the parser error buffer
#define BVHParserError(par, fmt, ...) \
    snprintf((par)->err, BVH_PARSER_ERR_MAX, "%s:%i:%i: error: " fmt, (par)->filename.c_str(), (par)->row_view, (par)->col, ##__VA_ARGS__)

//----------------------------------------------------------------------------------
// BVH File Data
//----------------------------------------------------------------------------------


static inline void BVHJointDataInit(BVHJointData* data)
{
    data->parent = -1;
    data->name.clear();
    data->offset = Vector3{ 0.0f, 0.0f, 0.0f };
    data->channelCount = 0;
    data->endSite = false;
}

static inline void BVHJointDataRename(BVHJointData* data, const char* name)
{
    data->name = name ? name : "";
}

static inline void BVHJointDataFree(BVHJointData* /*data*/)
{
    // no-op: std::string will clean up automatically
}


static inline void BVHDataInit(BVHData* bvh)
{
    bvh->jointCount = 0;
    bvh->joints.clear();
    bvh->frameCount = 0;
    bvh->channelCount = 0;
    bvh->frameTime = 0.0f;
    bvh->motionData.clear();
}

static inline void BVHDataFree(BVHData* bvh)
{
    // let destructors handle strings and vectors; keep explicit cleanup for clarity
    for (int i = 0; i < bvh->jointCount; i++)
    {
        BVHJointDataFree(&bvh->joints[i]);
    }
    bvh->joints.clear();
    bvh->motionData.clear();
    bvh->jointCount = 0;
    bvh->frameCount = 0;
    bvh->channelCount = 0;
}

// Add a joint and return its index
static inline int BVHDataAddJoint(BVHData* bvh)
{
    BVHJointData j;
    BVHJointDataInit(&j);
    bvh->joints.push_back(j);
    bvh->jointCount = (int)bvh->joints.size();
    return bvh->jointCount - 1;
}

//----------------------------------------------------------------------------------
// BVH Parsing Functions
//----------------------------------------------------------------------------------

// Parse any whitespace
static void BVHParseWhitespace(BVHParser* par)
{
    while (BVHParserOneOf(par, " \r\t\v")) { BVHParserInc(par); }
}

// Parse the given string (in a non-case sensitive way). I've found that in practice
// many BVH files don't respect case sensitivity so parsing any keywords in a non-case
// sensitive way seems safer.
static bool BVHParseString(BVHParser* par, const char* string)
{
    if (BVHParserStartsWithCaseless(par, string))
    {
        BVHParserAdvance(par, (int)strlen(string));
        return true;
    }
    else
    {
        BVHParserError(par, "expected '%s' at '%s'", string, BVHParserCharName(BVHParserPeek(par)));
        return false;
    }
}

// Parse any whitespace followed by a newline
static bool BVHParseNewline(BVHParser* par)
{
    BVHParseWhitespace(par);

    if (BVHParserMatch(par, '\n'))
    {
        BVHParserInc(par);
        BVHParseWhitespace(par);
        return true;
    }
    else
    {
        BVHParserError(par, "expected newline at '%s'", BVHParserCharName(BVHParserPeek(par)));
        return false;
    }
}

// Parse any whitespace and then an identifier for the name of a joint
static bool BVHParseJointName(BVHJointData* jnt, BVHParser* par)
{
    BVHParseWhitespace(par);

    char buffer[256];
    int chrnum = 0;
    while (chrnum < 255 && BVHParserOneOf(par,
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-."))
    {
        buffer[chrnum] = BVHParserPeek(par);
        chrnum++;
        BVHParserInc(par);
    }
    buffer[chrnum] = '\0';

    if (chrnum > 0)
    {
        BVHJointDataRename(jnt, buffer);
        BVHParseWhitespace(par);
        return true;
    }
    else
    {
        BVHParserError(par, "expected joint name at '%s'", BVHParserCharName(BVHParserPeek(par)));
        return false;
    }
}

// Parse a float value
static bool BVHParseFloat(float* out, BVHParser* par)
{
    BVHParseWhitespace(par);

    char* end;
    errno = 0;
    (*out) = (float)strtod(par->data + par->offset, &end);

    if (errno == 0)
    {
        BVHParserAdvance(par, (int)(end - (par->data + par->offset)));
        return true;
    }
    else
    {
        BVHParserError(par, "expected float at '%s'", BVHParserCharName(BVHParserPeek(par)));
        return false;
    }
}

// Parse an integer value
static bool BVHParseInt(int* out, BVHParser* par)
{
    BVHParseWhitespace(par);

    char* end;
    errno = 0;
    (*out) = (int)strtol(par->data + par->offset, &end, 10);

    if (errno == 0)
    {
        BVHParserAdvance(par, (int)(end - (par->data + par->offset)));
        return true;
    }
    else
    {
        BVHParserError(par, "expected integer at '%s'", BVHParserCharName(BVHParserPeek(par)));
        return false;
    }
}

// Parse the "joint offset" part of the BVH File
static bool BVHParseJointOffset(BVHJointData* jnt, BVHParser* par)
{
    if (!BVHParseString(par, "OFFSET")) { return false; }
    if (!BVHParseFloat(&jnt->offset.x, par)) { return false; }
    if (!BVHParseFloat(&jnt->offset.y, par)) { return false; }
    if (!BVHParseFloat(&jnt->offset.z, par)) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    return true;
}

// Parse a channel type and return it in "channel"
static bool BVHParseChannelEnum(
    char* channel,
    BVHParser* par,
    const char* channelName,
    char channelValue)
{
    BVHParseWhitespace(par);
    if (!BVHParseString(par, channelName)) { return false; }
    BVHParseWhitespace(par);
    *channel = channelValue;
    return true;
}

// Parse a channel type and return it in "channel"
static bool BVHParseChannel(char* channel, BVHParser* par)
{
    BVHParseWhitespace(par);

    if (BVHParserPeek(par) == '\0')
    {
        BVHParserError(par, "expected channel at end of file");
        return false;
    }

    // Here we are safe to peek forward an extra character since we've already
    // checked the current character is not the null terminator.

    if (BVHParserPeek(par) == 'X' && BVHParserPeekForward(par, 1) == 'p')
    {
        return BVHParseChannelEnum(channel, par, "Xposition", CHANNEL_X_POSITION);
    }

    if (BVHParserPeek(par) == 'Y' && BVHParserPeekForward(par, 1) == 'p')
    {
        return BVHParseChannelEnum(channel, par, "Yposition", CHANNEL_Y_POSITION);
    }

    if (BVHParserPeek(par) == 'Z' && BVHParserPeekForward(par, 1) == 'p')
    {
        return BVHParseChannelEnum(channel, par, "Zposition", CHANNEL_Z_POSITION);
    }

    if (BVHParserPeek(par) == 'X' && BVHParserPeekForward(par, 1) == 'r')
    {
        return BVHParseChannelEnum(channel, par, "Xrotation", CHANNEL_X_ROTATION);
    }

    if (BVHParserPeek(par) == 'Y' && BVHParserPeekForward(par, 1) == 'r')
    {
        return BVHParseChannelEnum(channel, par, "Yrotation", CHANNEL_Y_ROTATION);
    }

    if (BVHParserPeek(par) == 'Z' && BVHParserPeekForward(par, 1) == 'r')
    {
        return BVHParseChannelEnum(channel, par, "Zrotation", CHANNEL_Z_ROTATION);
    }

    BVHParserError(par, "expected channel type");
    return false;
}

// Parse the "channels" part of the BVH file format
static bool BVHParseJointChannels(BVHJointData* jnt, BVHParser* par)
{
    if (!BVHParseString(par, "CHANNELS")) { return false; }
    if (!BVHParseInt(&jnt->channelCount, par)) { return false; }

    for (int i = 0; i < jnt->channelCount; i++)
    {
        if (!BVHParseChannel(&jnt->channels[i], par)) { return false; }
    }

    if (!BVHParseNewline(par)) { return false; }

    return true;
}

// Parse a joint in the BVH file format
static bool BVHParseJoints(BVHData* bvh, int parent, BVHParser* par)
{
    while (BVHParserOneOf(par, "JEje")) // Either "JOINT" or "End Site"
    {
        int j = BVHDataAddJoint(bvh);
        bvh->joints[j].parent = parent;

        if (BVHParserMatch(par, 'J'))
        {
            if (!BVHParseString(par, "JOINT")) { return false; }
            if (!BVHParseJointName(&bvh->joints[j], par)) { return false; }
            if (!BVHParseNewline(par)) { return false; }
            if (!BVHParseString(par, "{")) { return false; }
            if (!BVHParseNewline(par)) { return false; }
            if (!BVHParseJointOffset(&bvh->joints[j], par)) { return false; }
            if (!BVHParseJointChannels(&bvh->joints[j], par)) { return false; }
            if (!BVHParseJoints(bvh, j, par)) { return false; }
            if (!BVHParseString(par, "}")) { return false; }
            if (!BVHParseNewline(par)) { return false; }
        }
        else if (BVHParserMatch(par, 'E'))
        {
            bvh->joints[j].endSite = true;

            if (!BVHParseString(par, "End Site")) { return false; }
            BVHJointDataRename(&bvh->joints[j], "End Site");
            if (!BVHParseNewline(par)) { return false; }
            if (!BVHParseString(par, "{")) { return false; }
            if (!BVHParseNewline(par)) { return false; }
            if (!BVHParseJointOffset(&bvh->joints[j], par)) { return false; }
            if (!BVHParseString(par, "}")) { return false; }
            if (!BVHParseNewline(par)) { return false; }
        }
    }

    return true;
}

// Parse the frame count
static bool BVHParseFrames(BVHData* bvh, BVHParser* par)
{
    if (!BVHParseString(par, "Frames:")) { return false; }
    if (!BVHParseInt(&bvh->frameCount, par)) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    return true;
}

// Parse the frame time
static bool BVHParseFrameTime(BVHData* bvh, BVHParser* par)
{
    if (!BVHParseString(par, "Frame Time:")) { return false; }
    if (!BVHParseFloat(&bvh->frameTime, par)) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    if (bvh->frameTime == 0.0f) { bvh->frameTime = 1.0f / 60.0f; }
    return true;
}

// Parse the motion data part of the BVH file format
static bool BVHParseMotionData(BVHData* bvh, BVHParser* par)
{
    int channelCount = 0;
    for (int i = 0; i < bvh->jointCount; i++)
    {
        channelCount += bvh->joints[i].channelCount;
    }

    bvh->channelCount = channelCount;
    // resize motionData to hold all frames*channels
    bvh->motionData.clear();
    bvh->motionData.resize((size_t)bvh->frameCount * (size_t)channelCount);

    for (int i = 0; i < bvh->frameCount; i++)
    {
        for (int j = 0; j < channelCount; j++)
        {
            if (!BVHParseFloat(&bvh->motionData[i * channelCount + j], par)) { return false; }
        }

        if (!BVHParseNewline(par)) { return false; }
    }

    return true;
}

// Parse the entire BVH file format
static bool BVHParse(BVHData* bvh, BVHParser* par)
{
    // Hierarchy Data

    if (!BVHParseString(par, "HIERARCHY")) { return false; }
    if (!BVHParseNewline(par)) { return false; }

    int j = BVHDataAddJoint(bvh);

    if (!BVHParseString(par, "ROOT")) { return false; }
    if (!BVHParseJointName(&bvh->joints[j], par)) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    if (!BVHParseString(par, "{")) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    if (!BVHParseJointOffset(&bvh->joints[j], par)) { return false; }
    if (!BVHParseJointChannels(&bvh->joints[j], par)) { return false; }
    if (!BVHParseJoints(bvh, j, par)) { return false; }
    if (!BVHParseString(par, "}")) { return false; }
    if (!BVHParseNewline(par)) { return false; }

    // Motion Data

    if (!BVHParseString(par, "MOTION")) { return false; }
    if (!BVHParseNewline(par)) { return false; }
    if (!BVHParseFrames(bvh, par)) { return false; }
    if (!BVHParseFrameTime(bvh, par)) { return false; }
    if (!BVHParseMotionData(bvh, par)) { return false; }

    return true;
}

// Load the given file and parse the contents as a BVH file.
static bool BVHDataLoad(BVHData* bvh, const char* filename, char* errMsg, int errMsgSize)
{
    // Read file Contents

    FILE* f = fopen(filename, "rb");

    if (f == NULL)
    {
        snprintf(errMsg, errMsgSize, "Error: Could not find file '%s'\n", filename);
        return false;
    }

    fseek(f, 0, SEEK_END);
    long int length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = (char*)malloc((size_t)length + 1);
    fread(buffer, 1, (size_t)length, f);
    buffer[length] = '\n';
    fclose(f);

    // Free and re-init in case we are re-using an old buffer
    BVHDataFree(bvh);
    BVHDataInit(bvh);

    // Parse BVH
    BVHParser par;
    BVHParserInit(&par, filename, buffer);
    bool result = BVHParse(bvh, &par);

    // Free contents and return result
    free(buffer);

    if (!result)
    {
        snprintf(errMsg, errMsgSize, "Error: Could not parse BVH file:\n    %s", par.err);
    }
    else
    {
        errMsg[0] = '\0';
        printf("INFO: parsed '%s' successfully\n", filename);
    }

    return result;
}

// =============================================================================
// BVH Writer - Save BVHData to file
// =============================================================================

// Helper to find children of a joint
static void BVHDataGetChildren(const BVHData* bvh, int parentIdx, int* children, int* childCount)
{
    *childCount = 0;
    for (int i = 0; i < bvh->jointCount; i++)
    {
        if (bvh->joints[i].parent == parentIdx)
        {
            children[(*childCount)++] = i;
        }
    }
}

// Write hierarchy recursively
static void BVHWriteHierarchy(FILE* f, const BVHData* bvh, int idx, int depth)
{
    const BVHJointData* j = &bvh->joints[idx];

    // Indentation
    for (int i = 0; i < depth * 2; i++) fputc(' ', f);

    if (j->endSite)
    {
        fprintf(f, "End Site\n");
        for (int i = 0; i < depth * 2; i++) fputc(' ', f);
        fprintf(f, "{\n");
        for (int i = 0; i < depth * 2 + 2; i++) fputc(' ', f);
        fprintf(f, "OFFSET %g %g %g\n", j->offset.x, j->offset.y, j->offset.z);
        for (int i = 0; i < depth * 2; i++) fputc(' ', f);
        fprintf(f, "}\n");
        return;
    }

    // ROOT or JOINT
    fprintf(f, "%s %s\n", (j->parent == -1) ? "ROOT" : "JOINT", j->name.c_str());
    for (int i = 0; i < depth * 2; i++) fputc(' ', f);
    fprintf(f, "{\n");

    // OFFSET
    for (int i = 0; i < depth * 2 + 2; i++) fputc(' ', f);
    fprintf(f, "OFFSET %g %g %g\n", j->offset.x, j->offset.y, j->offset.z);

    // CHANNELS
    if (j->channelCount > 0)
    {
        for (int i = 0; i < depth * 2 + 2; i++) fputc(' ', f);
        fprintf(f, "CHANNELS %d", j->channelCount);
        for (int c = 0; c < j->channelCount; c++)
        {
            const char* channelNames[] = {
                "Xposition", "Yposition", "Zposition",
                "Xrotation", "Yrotation", "Zrotation"
            };
            fprintf(f, " %s", channelNames[(int)j->channels[c]]);
        }
        fprintf(f, "\n");
    }

    // Children
    int children[256];
    int childCount;
    BVHDataGetChildren(bvh, idx, children, &childCount);
    for (int c = 0; c < childCount; c++)
    {
        BVHWriteHierarchy(f, bvh, children[c], depth + 1);
    }

    for (int i = 0; i < depth * 2; i++) fputc(' ', f);
    fprintf(f, "}\n");
}

// Save BVHData to a BVH file
static bool BVHDataSave(const BVHData* bvh, const char* filename, char* errMsg, int errMsgSize)
{
    FILE* f = fopen(filename, "w");
    if (f == NULL)
    {
        snprintf(errMsg, errMsgSize, "Error: Could not create file '%s'\n", filename);
        return false;
    }

    // Find root joint
    int rootIdx = -1;
    for (int i = 0; i < bvh->jointCount; i++)
    {
        if (bvh->joints[i].parent == -1 && !bvh->joints[i].endSite)
        {
            rootIdx = i;
            break;
        }
    }

    if (rootIdx < 0)
    {
        snprintf(errMsg, errMsgSize, "Error: No root joint found\n");
        fclose(f);
        return false;
    }

    // Write hierarchy
    fprintf(f, "HIERARCHY\n");
    BVHWriteHierarchy(f, bvh, rootIdx, 0);

    // Write motion data
    fprintf(f, "MOTION\n");
    fprintf(f, "Frames: %d\n", bvh->frameCount);
    fprintf(f, "Frame Time: %f\n", bvh->frameTime);

    for (int frame = 0; frame < bvh->frameCount; frame++)
    {
        for (int c = 0; c < bvh->channelCount; c++)
        {
            if (c > 0) fputc(' ', f);
            fprintf(f, "%g", bvh->motionData[frame * bvh->channelCount + c]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("INFO: Saved '%s' successfully\n", filename);
    return true;
}

//----------------------------------------------------------------------------------
// Joint Search Utilities
//----------------------------------------------------------------------------------

// Find a joint index by matching against a list of candidate names (case-insensitive).
// First tries exact match, then substring match.
static inline int FindJointIndexByNames(const BVHData* bvh, const std::vector<std::string>& candidates)
{
    // Exact match pass (case-insensitive)
    for (int j = 0; j < bvh->jointCount; ++j)
    {
        const std::string& name = bvh->joints[j].name;
        if (name.empty()) continue;
        std::string lname = ToLowerCopy(name.c_str());
        for (int k = 0; k < (int)candidates.size(); ++k)
        {
            if (lname == candidates[k])
            {
                return j;
            }
        }
    }

    // Substring fallback pass (case-insensitive)
    for (int j = 0; j < bvh->jointCount; ++j)
    {
        const std::string& name = bvh->joints[j].name;
        if (name.empty()) continue;
        std::string lname = ToLowerCopy(name.c_str());
        for (int k = 0; k < (int)candidates.size(); ++k)
        {
            if (StrContainsCaseInsensitive(lname, candidates[k]))
            {
                return j;
            }
        }
    }

    return -1;
}