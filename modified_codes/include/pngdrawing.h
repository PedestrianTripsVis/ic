#pragma once

#include "include/glutwrapper.h"
#include <png.h>

#include <iostream>
#include <vector>
#include <setjmp.h>
#include <unistd.h>


struct PngDrawing {
    bool canDraw = false,
         textureLoaded = false;
    png_byte *imageData = NULL;
    float border = 0.0,
          scale = 1;
    const int headerSize = 8;
    int bitDepth, 
        colorType;
    png_uint_32 width, 
                height;

    GLuint texture[1];

    ~PngDrawing() {
        if (imageData != NULL) free(imageData);
    };
  
    PngDrawing(const char* filename);
    PngDrawing(const char* filename, float _scale, float _border);


    bool loadImageData(const char* filename);
    bool loadTexture();


    void draw();
};