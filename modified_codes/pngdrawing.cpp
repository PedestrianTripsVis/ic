#include "include/pngdrawing.h"

int getGLColorType(int colorType);
void loadTexture(PngDrawing& _png);

PngDrawing::PngDrawing(const char* filename) {
    canDraw = loadImageData(filename);
    textureLoaded = loadTexture();
}

PngDrawing::PngDrawing(const char* filename, float _scale, float _border) {
    scale = _scale;
    border = _border;
    canDraw = loadImageData(filename);
    textureLoaded = loadTexture();
}

bool PngDrawing::loadImageData(const char* filename) {
    FILE *fp{fopen(filename, "r")};
    if (!fp) return false;

    png_byte header[headerSize];
    if (fread(header, sizeof(png_byte), headerSize, fp) != headerSize) {
        fclose(fp);
        return false;
    }
    
    if (png_sig_cmp(header, 0, headerSize)) {
        fclose(fp);
        return false;
    }

    auto png_ptr{png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL)};
    if (!png_ptr) {
        fclose(fp);
        return false;
    }

    auto info_ptr{png_create_info_struct(png_ptr)};
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL );
        fclose(fp);
        return false;
    }

    auto end_info{png_create_info_struct(png_ptr)};
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, headerSize);
    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bitDepth, &colorType, NULL, NULL, NULL);
    png_read_update_info( png_ptr, info_ptr );


    colorType = getGLColorType(colorType);
    if (colorType == -1) {
        printf("PngDrawing Error: Color type %d not supported.\n", colorType);
        png_destroy_read_struct(&png_ptr,&info_ptr,&end_info);
        fclose(fp);
        return false;
    }

    auto rowBytes{png_get_rowbytes(png_ptr, info_ptr)};
    auto rowPointers{(png_bytepp) malloc(height * sizeof(png_bytep))};
    imageData =(png_bytep) realloc(imageData, rowBytes * height* sizeof(png_byte));
    
    for (int i = 0; i < height; ++i) rowPointers[height - 1 - i] = imageData + i * rowBytes;

    png_read_image(png_ptr, rowPointers);
    
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    free(rowPointers);
    fclose(fp);
    
    return true;
}

bool PngDrawing::loadTexture() {
    if (!canDraw) {
        printf("PngDrawing Error: Could not load texture\n");
        return false;
    }

    glGenTextures(1, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, texture[0]);

    return true;
}

void PngDrawing::draw() {
    if (!canDraw || !textureLoaded) {
        printf("PngDrawing Error: Could not draw texture\n");
       return;
    }


    glBindTexture(GL_TEXTURE_2D, texture[0]); 
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, colorType, GL_UNSIGNED_BYTE, imageData);  

    glDisable(GL_BLEND);
    glEnable(GL_TEXTURE_2D);
    
    glBegin(GL_QUADS);
    glTexCoord2f( 1.0f, 0.0f );
    glVertex3f( 1.0f, -1.0, 0.0f );
    glTexCoord2f( 0.0f, 0.0f );
    glVertex3f( -1.0f, -1.0f, 0.0f );
    glTexCoord2f( 0.0f, 1.0f );
    glVertex3f( -1.0f, 1.0f, 0.0f );
    glTexCoord2f( 1.0f, 1.0f );
    glVertex3f( 1.0f, 1.0f, 0.0f );
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

int getGLColorType(int colorType) {
    switch (colorType) {
        case PNG_COLOR_TYPE_RGBA:
            return GL_RGBA;

        case PNG_COLOR_TYPE_RGB:
            return GL_RGB;
    }
    
    return -1;
}

