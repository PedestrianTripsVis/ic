#include "include/shapetopointdrawing.h"
#include <cstring>

ShapeToPointDrawing::ShapeToPointDrawing(const char* shapePath, int fboSize, float border, MapDrawing* map){
  this->fboSize = fboSize;
  this->border = border;
  this->map = map;
  canDraw = loadPoints(shapePath);
}


bool ShapeToPointDrawing::loadPoints(const char* shapePath)
{
  if(canDraw) return true; //Already loaded points, do not do it again
  if(shapePath == 0 || shapePath == NULL) return false; // no file to read, return false

  SHPHandle myHandler = SHPOpen(shapePath, "rb");
  dbfHandler = DBFOpen(shapePath, "rb");
	if(myHandler == NULL || dbfHandler == NULL) return false;

  SHPGetInfo(myHandler, &nEntities, &pnShapeType, padfMinBound, padfMaxBound);
  shapeOffsets.assign(nEntities, 0);

  float xMin = map->minX();
  float yMin = map->minY();

  for(int T=0; T<nEntities; T++){
    SHPObject *obj = SHPReadObject(myHandler, T);
    shapeOffsets[T] = obj->nVertices;

    float x, y, z = 0;
    float rangeX = map->getRangeX();
    float rangeY = map->getRangeY();

    // Get the scale based on max range X or Y axes
    if (rangeX>rangeY)
      scale  = (1-border)*fboSize/rangeX;
    else
      scale  = (1-border)*fboSize/rangeY;

    float xTranslation = (fboSize-scale*rangeX)/2;//Our points fall into 0,1 quadrant, we need to
    float yTranslation = (fboSize-scale*rangeY)/2;//make a translation in both axes to center in -1,1

    for(int i=0; i < obj->nVertices; i++) {
      // Vertex points to be draw are made of 3 float elements (X, Y, Z)

      x = obj->padfX[i] - xMin;
      y = obj->padfY[i] - yMin;

			//Scale points and make a translation to center points in between [-1,1]
      x = x*scale+xTranslation;
      y = y*scale+yTranslation;

      //Add X, Y, Z coordinates to points data array
      shapePoints.push_back(x);
      shapePoints.push_back(y);

    }
  }
  SHPClose(myHandler);

  return true;
}

void ShapeToPointDrawing::draw(int point_size, float color[4])
{
  int last = 0;
  if(this->canDraw) {
    glEnableClientState(GL_VERTEX_ARRAY);                           //Enable the GL vertex and color arrays.
    glVertexPointer(2, GL_FLOAT, 0, shapePoints.data());
    glPointSize(point_size);  
    glEnable(GL_BLEND);
    glColor4f(0.0,0.0,0.0, 0.8);

    for(int j=0; j<nEntities; j++) {
      glColor4fv(color);
      glDrawArrays(GL_POINTS, last, shapeOffsets[j]); 
      last += shapeOffsets[j];
    }

    glDisableClientState(GL_VERTEX_ARRAY);                           //Enable the GL vertex and color arrays.
  }
}
