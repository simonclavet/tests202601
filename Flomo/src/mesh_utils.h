#pragma once

// =============================================================================
// Mesh Utilities - OBJ loading and embedded mesh data
// =============================================================================
//
// Contains helper functions for loading OBJ meshes from memory,
// plus embedded mesh data for primitive shapes like capsules.
//

#include "raylib.h"
#include "raymath.h"

extern "C" {
#include "external/tinyobj_loader_c.h"
}

#include <cstring>

// =============================================================================
// Embedded Capsule OBJ
// =============================================================================
// A simple capsule mesh with rounded ends, oriented along X axis.

static const char* CAPSULE_OBJ_DATA = "\
v 0.82165808 -0.82165808 -1.0579772e-18\nv 0.82165808 -0.58100000 0.58100000\n\
v 0.82165808 8.7595780e-17 0.82165808\nv 0.82165808 0.58100000 0.58100000\n\
v 0.82165808 0.82165808 9.9566116e-17\nv 0.82165808 0.58100000 -0.58100000\n\
v 0.82165808 2.8884397e-16 -0.82165808\nv 0.82165808 -0.58100000 -0.58100000\n\
v -0.82165808 -0.82165808 -1.0579772e-18\nv -0.82165808 -0.58100000 0.58100000\n\
v -0.82165808 -1.3028313e-17 0.82165808\nv -0.82165808 0.58100000 0.58100000\n\
v -0.82165808 0.82165808 9.9566116e-17\nv -0.82165808 0.58100000 -0.58100000\n\
v -0.82165808 1.8821987e-16 -0.82165808\nv -0.82165808 -0.58100000 -0.58100000\n\
v 1.16200000 1.5874776e-16 -1.0579772e-18\nv -1.16200000 1.6443801e-17 -1.0579772e-18\n\
v -9.1030792e-3 -1.15822938 -1.0579772e-18\nv 9.1030792e-3 -1.15822938 -1.0579772e-18\n\
v 9.1030792e-3 -0.81899185 0.81899185\nv -9.1030792e-3 -0.81899185 0.81899185\n\
v 9.1030792e-3 1.7232088e-17 1.15822938\nv -9.1030792e-3 1.6117282e-17 1.15822938\n\
v 9.1030792e-3 0.81899185 0.81899185\nv -9.1030792e-3 0.81899185 0.81899185\n\
v 9.1030792e-3 1.15822938 1.4078421e-16\nv -9.1030792e-3 1.15822938 1.4078421e-16\n\
v 9.1030792e-3 0.81899185 -0.81899185\nv -9.1030792e-3 0.81899185 -0.81899185\n\
v 9.1030792e-3 3.0091647e-16 -1.15822938\nv -9.1030792e-3 2.9980166e-16 -1.15822938\n\
v 9.1030792e-3 -0.81899185 -0.81899185\nv -9.1030792e-3 -0.81899185 -0.81899185\n\
vn 0.71524683 -0.69887193 -2.5012597e-16\nvn 0.61185516 -0.55930013 0.55930013\n\
vn 0.71524683 0.0000000e+0 0.69887193\nvn 0.61185516 0.55930013 0.55930013\n\
vn 0.71524683 0.69887193 1.5632873e-17\nvn 0.61185516 0.55930013 -0.55930013\n\
vn 0.71524683 6.2531494e-17 -0.69887193\nvn 0.61185516 -0.55930013 -0.55930013\n\
vn -0.71524683 -0.69887193 -2.5012597e-16\nvn -0.61185516 -0.55930013 0.55930013\n\
vn -0.71524683 0.0000000e+0 0.69887193\nvn -0.61185516 0.55930013 0.55930013\n\
vn -0.71524683 0.69887193 4.6898620e-17\nvn -0.61185516 0.55930013 -0.55930013\n\
vn -0.71524683 4.6898620e-17 -0.69887193\nvn -0.61185516 -0.55930013 -0.55930013\n\
vn 1.00000000 1.5208752e-17 -2.6615316e-17\nvn -1.00000000 -1.5208752e-17 2.2813128e-17\n\
vn -0.19614758 -0.98057439 -2.2848712e-16\nvn 0.26047011 -0.96548191 -2.4273177e-16\n\
vn 0.13072302 -0.70103905 0.70103905\nvn -0.19614758 -0.69337080 0.69337080\n\
vn 0.22349711 5.9825845e-2 0.97286685\nvn -0.22349711 -5.9825845e-2 0.97286685\n\
vn 0.15641931 0.75510180 0.63667438\nvn -0.15641931 0.63667438 0.75510180\n\
vn 0.22349711 0.97286685 -5.9825845e-2\nvn -0.22349711 0.97286685 5.9825845e-2\n\
vn 0.15641931 0.63667438 -0.75510180\nvn -0.15641931 0.75510180 -0.63667438\n\
vn 0.22349711 -5.9825845e-2 -0.97286685\nvn -0.22349711 5.9825845e-2 -0.97286685\n\
vn 0.15641931 -0.75510180 -0.63667438\nvn -0.15641931 -0.63667438 -0.75510180\n\
f 1//1 17//17 2//2\nf 1//1 20//20 8//8\nf 2//2 17//17 3//3\nf 2//2 20//20 1//1\n\
f 2//2 23//23 21//21\nf 3//3 17//17 4//4\nf 3//3 23//23 2//2\nf 4//4 17//17 5//5\n\
f 4//4 23//23 3//3\nf 4//4 27//27 25//25\nf 5//5 17//17 6//6\nf 5//5 27//27 4//4\n\
f 6//6 17//17 7//7\nf 6//6 27//27 5//5\nf 6//6 31//31 29//29\nf 7//7 17//17 8//8\n\
f 7//7 31//31 6//6\nf 8//8 17//17 1//1\nf 8//8 20//20 33//33\nf 8//8 31//31 7//7\n\
f 9//9 18//18 16//16\nf 9//9 19//19 10//10\nf 10//10 18//18 9//9\nf 10//10 19//19 22//22\n\
f 10//10 24//24 11//11\nf 11//11 18//18 10//10\nf 11//11 24//24 12//12\nf 12//12 18//18 11//11\n\
f 12//12 24//24 26//26\nf 12//12 28//28 13//13\nf 13//13 18//18 12//12\nf 13//13 28//28 14//14\n\
f 14//14 18//18 13//13\nf 14//14 28//28 30//30\nf 14//14 32//32 15//15\nf 15//15 18//18 14//14\n\
f 15//15 32//32 16//16\nf 16//16 18//18 15//15\nf 16//16 19//19 9//9\nf 16//16 32//32 34//34\n\
f 19//19 33//33 20//20\nf 20//20 21//21 19//19\nf 21//21 20//20 2//2\nf 21//21 24//24 22//22\n\
f 22//22 19//19 21//21\nf 22//22 24//24 10//10\nf 23//23 26//26 24//24\nf 24//24 21//21 23//23\n\
f 25//25 23//23 4//4\nf 25//25 28//28 26//26\nf 26//26 23//23 25//25\nf 26//26 28//28 12//12\n\
f 27//27 30//30 28//28\nf 28//28 25//25 27//27\nf 29//29 27//27 6//6\nf 29//29 32//32 30//30\n\
f 30//30 27//27 29//29\nf 30//30 32//32 14//14\nf 31//31 34//34 32//32\nf 32//32 29//29 31//31\n\
f 33//33 19//19 34//34\nf 33//33 31//31 8//8\nf 34//34 19//19 16//16\nf 34//34 31//31 33//33";

// =============================================================================
// OBJ Loading from Memory
// =============================================================================

// Load an OBJ model from a string in memory (uses tinyobj_loader_c).
// Falls back to a simple cylinder if parsing fails.
static Model LoadOBJFromMemory(const char* fileText)
{
    Model model = { 0 };

    tinyobj_attrib_t attrib = { 0 };
    tinyobj_shape_t* meshes = NULL;
    unsigned int meshCount = 0;

    tinyobj_material_t* materials = NULL;
    unsigned int materialCount = 0;

    if (fileText != NULL)
    {
        const unsigned int dataSize = (unsigned int)strlen(fileText);
        const unsigned int flags = TINYOBJ_FLAG_TRIANGULATE;

        tinyobj_parse_obj(&attrib, &meshes, &meshCount, &materials, &materialCount,
                          fileText, dataSize, flags);

        model.meshCount = 1;
        model.meshes = (Mesh*)RL_CALLOC(model.meshCount, sizeof(Mesh));
        model.meshMaterial = (int*)RL_CALLOC(model.meshCount, sizeof(int));

        // Count faces for each material
        int* matFaces = (int*)RL_CALLOC(model.meshCount, sizeof(int));
        matFaces[0] = attrib.num_faces;

        // Running counts for building meshes
        int* vCount = (int*)RL_CALLOC(model.meshCount, sizeof(int));
        int* vtCount = (int*)RL_CALLOC(model.meshCount, sizeof(int));
        int* vnCount = (int*)RL_CALLOC(model.meshCount, sizeof(int));
        int* faceCount = (int*)RL_CALLOC(model.meshCount, sizeof(int));

        // Allocate space for each material mesh
        for (int mi = 0; mi < model.meshCount; mi++)
        {
            model.meshes[mi].vertexCount = matFaces[mi] * 3;
            model.meshes[mi].triangleCount = matFaces[mi];
            model.meshes[mi].vertices = (float*)RL_CALLOC(model.meshes[mi].vertexCount * 3, sizeof(float));
            model.meshes[mi].texcoords = (float*)RL_CALLOC(model.meshes[mi].vertexCount * 2, sizeof(float));
            model.meshes[mi].normals = (float*)RL_CALLOC(model.meshes[mi].vertexCount * 3, sizeof(float));
            model.meshMaterial[mi] = mi;
        }

        // Scan through faces and fill mesh data
        for (unsigned int af = 0; af < attrib.num_faces; af++)
        {
            int mm = attrib.material_ids[af];
            if (mm == -1) { mm = 0; }

            tinyobj_vertex_index_t idx0 = attrib.faces[3 * af + 0];
            tinyobj_vertex_index_t idx1 = attrib.faces[3 * af + 1];
            tinyobj_vertex_index_t idx2 = attrib.faces[3 * af + 2];

            // Vertices
            for (int v = 0; v < 3; v++) {
                model.meshes[mm].vertices[vCount[mm] + v] = attrib.vertices[idx0.v_idx * 3 + v];
            }
            vCount[mm] += 3;
            for (int v = 0; v < 3; v++) {
                model.meshes[mm].vertices[vCount[mm] + v] = attrib.vertices[idx1.v_idx * 3 + v];
            }
            vCount[mm] += 3;
            for (int v = 0; v < 3; v++) {
                model.meshes[mm].vertices[vCount[mm] + v] = attrib.vertices[idx2.v_idx * 3 + v];
            }
            vCount[mm] += 3;

            // Texcoords (flip Y for raylib's coordinate system)
            if (attrib.num_texcoords > 0)
            {
                model.meshes[mm].texcoords[vtCount[mm] + 0] = attrib.texcoords[idx0.vt_idx * 2 + 0];
                model.meshes[mm].texcoords[vtCount[mm] + 1] = 1.0f - attrib.texcoords[idx0.vt_idx * 2 + 1];
                vtCount[mm] += 2;
                model.meshes[mm].texcoords[vtCount[mm] + 0] = attrib.texcoords[idx1.vt_idx * 2 + 0];
                model.meshes[mm].texcoords[vtCount[mm] + 1] = 1.0f - attrib.texcoords[idx1.vt_idx * 2 + 1];
                vtCount[mm] += 2;
                model.meshes[mm].texcoords[vtCount[mm] + 0] = attrib.texcoords[idx2.vt_idx * 2 + 0];
                model.meshes[mm].texcoords[vtCount[mm] + 1] = 1.0f - attrib.texcoords[idx2.vt_idx * 2 + 1];
                vtCount[mm] += 2;
            }

            // Normals
            if (attrib.num_normals > 0)
            {
                for (int v = 0; v < 3; v++) {
                    model.meshes[mm].normals[vnCount[mm] + v] = attrib.normals[idx0.vn_idx * 3 + v];
                }
                vnCount[mm] += 3;
                for (int v = 0; v < 3; v++) {
                    model.meshes[mm].normals[vnCount[mm] + v] = attrib.normals[idx1.vn_idx * 3 + v];
                }
                vnCount[mm] += 3;
                for (int v = 0; v < 3; v++) {
                    model.meshes[mm].normals[vnCount[mm] + v] = attrib.normals[idx2.vn_idx * 3 + v];
                }
                vnCount[mm] += 3;
            }
        }

        model.materialCount = 1;
        model.materials = (Material*)RL_CALLOC(model.materialCount, sizeof(Material));
        model.materials[0] = LoadMaterialDefault();

        tinyobj_attrib_free(&attrib);
        tinyobj_shapes_free(meshes, meshCount);
        tinyobj_materials_free(materials, materialCount);

        RL_FREE(matFaces);
        RL_FREE(vCount);
        RL_FREE(vtCount);
        RL_FREE(vnCount);
        RL_FREE(faceCount);
    }

    model.transform = MatrixIdentity();

    // Upload vertex data to GPU
    for (int i = 0; i < model.meshCount; i++)
    {
        UploadMesh(&model.meshes[i], false);
    }

    // Fallback if loading failed
    if (model.meshCount == 0 || model.meshes == NULL)
    {
        TraceLog(LOG_ERROR, "Failed to load OBJ from memory, using fallback cylinder");
        model = LoadModelFromMesh(GenMeshCylinder(0.5f, 2.0f, 8));
    }

    return model;
}

// Convenience function to load the embedded capsule mesh
static Model LoadCapsuleModel()
{
    return LoadOBJFromMemory(CAPSULE_OBJ_DATA);
}
