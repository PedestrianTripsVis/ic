
#pragma once

#include "include/glwrapper.h"
#include "include/mapdrawing.h"

#include <vector>
#include <shapefil.h>
#include <iostream>

struct ShapeToPointDrawing
{
    // the program ID
    float scale = 1;
    float color[3] = {1.0, 0.0, 0.0}; // light-grey
    bool canDraw = false;
    int fboSize = 512;
    float border = 0.0;
    DBFHandle dbfHandler;
    MapDrawing* map = NULL;
    std::vector<float> shapePoints; //All points from all shapes
    std::vector<float> shapeOffsets; //Offset where each shape starts
    int nEntities; //Total number of shapes 
    int pnShapeType;
    double padfMinBound[4], padfMaxBound[4];

    // constructor reads and builds the shader
    ShapeToPointDrawing(const char* shapePath, int fboSize, float border, MapDrawing* map);

    bool loadPoints(const char* shapePath);

    void draw(int point_size, float color[4]);
};


