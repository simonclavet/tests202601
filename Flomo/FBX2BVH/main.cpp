#include "raylib.h"
#include "fbx_converter.h"
#include <iostream>

int main(int argc, char* argv[]) {
    // ----------------------------------------------------
    // Command-line mode: convert and exit
    // ----------------------------------------------------
    if (argc > 1) {
        std::string inputPath = argv[1];

        if (inputPath.size() < 4 || inputPath.substr(inputPath.size() - 4) != ".fbx") {
            // Try case-insensitive
            std::string ext = inputPath.substr(inputPath.size() - 4);
            for (auto& c : ext) c = (char)tolower(c);
            if (ext != ".fbx") {
                std::cerr << "Error: Please provide an .fbx file." << std::endl;
                return 1;
            }
        }

        ConverterConfig config;
        config.inputFile = inputPath;
        config.outputFile = inputPath + ".bvh";

        FBXtoBVH converter(config);

        if (!converter.Load()) {
            std::cerr << "Error: Failed to load FBX." << std::endl;
            return 1;
        }

        if (!converter.BuildSkeleton()) {
            std::cerr << "Error: Could not build skeleton." << std::endl;
            return 1;
        }

        converter.ExtractAnimations();
        converter.SaveBVH();

        return 0;
    }

    // ----------------------------------------------------
    // Raylib Initialization (GUI mode)
    // ----------------------------------------------------
    InitWindow(800, 450, "Raylib FBX to BVH Converter");
    SetTargetFPS(60);

    // ----------------------------------------------------
    // Converter Setup
    // ----------------------------------------------------
    std::string status = "Drag and drop an .FBX file...";

    while (!WindowShouldClose()) {
        // Handle Drag and Drop
        if (IsFileDropped()) {
            FilePathList droppedFiles = LoadDroppedFiles();

            if (droppedFiles.count > 0) {
                std::string inputPath = droppedFiles.paths[0];

                if (IsFileExtension(inputPath.c_str(), ".fbx")) {
                    status = "Converting: " + inputPath + "...";

                    // Run Conversion
                    ConverterConfig config;
                    config.inputFile = inputPath;
                    config.outputFile = inputPath + ".bvh"; // Simple rename
                    // Note: ufbx already converts to meters, no additional scaling needed

                    FBXtoBVH converter(config);

                    if (converter.Load()) {
                        if (converter.BuildSkeleton()) {
                            converter.ExtractAnimations();
                            converter.SaveBVH();
                            status = "Success! Saved to " + config.outputFile;
                        }
                        else {
                            status = "Error: Could not build skeleton.";
                        }
                    }
                    else {
                        status = "Error: Failed to load FBX.";
                    }
                }
                else {
                    status = "Error: Please drop an .fbx file.";
                }
            }
            UnloadDroppedFiles(droppedFiles);
        }

        // Draw
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText("FBX -> BVH Converter", 20, 20, 30, DARKGRAY);
        DrawText(status.c_str(), 20, 80, 20, LIGHTGRAY);

        if (status.find("Success") != std::string::npos) {
            DrawText("Check the folder where your FBX is located.", 20, 110, 15, GRAY);
        }
        EndDrawing();
    }

    CloseWindow();
    return 0;
}