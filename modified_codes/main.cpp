#include <GL/glew.h>

#include "include/cpubundling.h"
#include "include/gdrawing.h"
#include "include/glutwrapper.h"
#include "include/gluiwrapper.h"
#include <GL/glui.h>

#include <cuda_gl_interop.h>
#include <math.h>
#include <iostream>
#include <string>
#include <time.h>
#include <variant>
#define DBG(x) std::cout << #x << ": " <<  x << endl

#define TIME(call)   \
{                   \
   clock_t start, end; \
   start = clock();    \
   call;            \
   end = clock();    \
   std::cout << #call << " - " << (double)(end - start)/CLOCKS_PER_SEC << "s" << endl; \
}


using namespace std;


GraphDrawing*		gdrawing_orig;                                                      //Original drawing, as read from the input file. Never changed.
GraphDrawing*		gdrawing_bund;                                                      //Bundled drawing, computed from gdrawing_orig.
GraphDrawing*		gdrawing_final;                                                     //Final drawing, done by relaxation-interpolation between gdrawing_orig and gdrawing_bund.
CPUBundling*		bund;
static float		scale = 1;                                                          //Scaling of coords from input file to screen pixels
static float		start_time = 0.0;                                                          //Scaling of coords from input file to screen pixels
static float		end_time = 23.99;                                                          //Scaling of coords from input file to screen pixels
static int      departureHour = 0;
static int      departureMinutes = 0;
static int      arrivalHour = 23;
static int      arrivalMinutes = 59;
static float		transX = 0, transY = 0;                                             //Translation of bbox of coords from input file to screen window

static int			fboSize;                                                            //Size of window (pow of 2)
static float		relaxation = 0;                                                     //Interpolation between gdrawing_orig and gdrawing_bund (in [0,1])
static float        max_displacement = 0.2;                                             //Max displacement allowed for a bundled-edge sampling point w.r.t. gdrawing_orig
static int          displ_rel_edgelength = 1;                                           //If true, max_displacement is fraction of fboSize; else max_displacement is fraction of current edge-length
static float		shading_radius = 3;                                                 //Radius of kernel used to normalize density-map for shading computations (pixels)
static int			show_points	= 0;                                                    //Show edge sample points or not
static int			show_edges	= 1;                                                    //Show graph edges or not
static int			show_endpoints = 0;                                                 //Show edge endpoints or not
static int			gpu_bundling = 1;                                                   //1 = do GPU bundling; 0 = do CPU bundling
static int          auto_update = 0;                                                    //1 = rebundle upon any relevant parameter change; 0 = ask user to explicitly press 'Bundle 1x'
static int			density_estimation = 0;                                             //Type of GPU KDE density estimation: 0=exact (using atomic ops); 1=fast (no atomics)
static int			color_mode = GraphDrawing::RAINBOW;                                 //Color map used to map various data fields to color
static int			alpha_mode = GraphDrawing::ALPHA_CONSTANT;                          //How to map edge values to alpha
static int			polyline_style = 0;                                                 //Bundle using polyline style or not
static int			bundle_shape = 0;                                                   //Shape of bundles (0=FDEB, 1=HEB)
static int			tangent = false;                                                    //
static int			block_endpoints = 1;                                                //Do not allow edgepoints to move during bundling (typically what we want)
static int			use_density_alpha = 0;                                              //Modulate final drawing's alpha by edge-density map or not
static int			shading = 0;                                                        //Use shading of bundles or not
static int			shading_tube = 0;                                                   //Style of shading (0=tube, 1=Phong)
static float		dir_separation = 0;                                                 //Separate different-direction bundles in the 'tracks' style or not
static int			draw_background = 1;                                                //Draw background image under graph or not
static int			draw_rails = 0;                                                //Draw background image under graph or not
static int 			draw_png = 0;
static int      	print_screen = 0;
static const char*  save_filename = "CUBu_bundling.trl";                                //Name of text-file where to save bundling result
static float obj_pos[] = { 0.0, 0.0, 0.0 };

int  show_transport_types[20] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};                                               //Show all transport_types

static const float border = 0.0;

static GLUI*		glui;

enum
{
		UI_SHOW_BUNDLES,
		UI_BUNDLE_ITERATIONS,
		UI_BUNDLE_MS_ITERATIONS,
		UI_BUNDLE_KERNEL,
		UI_BUNDLE_MS_KERNEL,
		UI_BUNDLE_EDGERES,
		UI_BUNDLE_SMOOTH,
		UI_BUNDLE_SMOOTH_ITER,
		UI_BUNDLE_SPEED,
		UI_BUNDLE_DENS_ESTIM,
		UI_BUNDLE_COLOR_MODE,
		UI_BUNDLE_ALPHA_MODE,
		UI_BUNDLE_CPU_GPU,
        UI_BUNDLE_AUTO_UPDATE,
		UI_BUNDLE_SHAPE,
		UI_BUNDLE_SHOW_POINTS,
		UI_BUNDLE_SHOW_EDGES,
		UI_BUNDLE_SHOW_ENDPOINTS,
		UI_BUNDLE_LINEWIDTH,
		UI_BUNDLE_GLOBAL_ALPHA,
		UI_BUNDLE_BLOCK_ENDS,
		UI_BUNDLE_POLYLINE_STYLE,
		UI_BUNDLE_SMOOTH_ENDS,
		UI_BUNDLE_RELAXATION,
        UI_BUNDLE_CLAMP,
        UI_BUNDLE_CLAMP_ABS_REL,
		UI_BUNDLE_DIRECTIONAL,
		UI_BUNDLE_DIR_REPULSION,
		UI_BUNDLE_DENSITY_ALPHA,
		UI_BUNDLE_DIR_SEPARATION,
		UI_BUNDLE_USE_SHADING,
		UI_BUNDLE_SHADING_RADIUS,
		UI_BUNDLE_SHADING_TUBE,
		UI_BUNDLE_SHADING_AMBIENT,
		UI_BUNDLE_SHADING_DIFFUSE,
		UI_BUNDLE_SHADING_SPECULAR,
		UI_BUNDLE_SHADING_SPECULAR_SIZE,
		UI_BUNDLE_BACK_IMAGE,
		UI_BUNDLE_RAILS_IMAGE,
		UI_BUNDLE_BASEMAP_IMAGE,
        UI_SAMPLE,
        UI_BUNDLE,
        UI_COPY_DRAWING_TO_ORIG,
        UI_SCREENSHOT,
        UI_RESET,
        UI_SAVE,
		UI_QUIT,
    UI_SHOW_TRANSPORT,
    UI_TIME
};


void display_cb();
void buildGUI(int);
void bundle(int force_update=0);
void postprocess();
void PPMWriter(unsigned char *in, char *name, int dimx, int dimy);
void saveImage(int width, int height);
void save_screenshot();
int saveFile(const char *mode);

void save_screenshot() {
  	int width = glutGet(GLUT_WINDOW_WIDTH);
  	int height = glutGet(GLUT_WINDOW_HEIGHT);

	saveImage(width, height);
}

void PPMWriter(unsigned char *in,char *name,int dimx, int dimy)
{
  	FILE *fp = fopen(name, "wb"); /* b - binary mode */
	if (!fp) {
		printf("Error: cannot open file path=\'%s\'\n", name);
		return;
	}
  	
	(void) fprintf(fp, "P6 %d %d 255 ", dimx, dimy);
	printf("screenshot res: %d %d\n", dimx, dimy);	
	for (int i = (dimy*dimx - dimx)*3; i >= 0; i -= (dimx*3)) {
		for (int j = 0; j < (dimx * 3); j += 3) {
			static unsigned char color[3];
    		color[0] = in[i+j];
    		color[1] = in[i+j+1];
    		color[2] = in[i+j+2];
    		fwrite(color, 1, 3, fp);  
		}	
	}

  	(void) fclose(fp);
}

void saveImage(int width, int height)
{	
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * 3 * width * height);
	
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, image);
   
    /* Saves screenshot with a timestamp*/
    time_t timer = time(NULL);
    struct tm *timeinfo = localtime(&timer);
    char buffer[100];
    strftime(buffer, 100, "captures/screenshot-%F-%H-%M-%S.ppm", timeinfo);

    PPMWriter(image, buffer, width, height);
	free(image);
}

int main(int argc,char **argv)
{
	char* graphfile = 0;
  	char* mapfile = 0;
	char* basemapfile = 0;
	fboSize   = 512;
	bool only_endpoints = false;
	bool savefile = false;
	int  max_edges = 0;

	for (int ar=1;ar<argc;++ar)
	{
		string opt = argv[ar];
		if (opt=="-f")											//Input file name:
		{
			++ar;
			graphfile = argv[ar];
		}
		else if (opt=="-i")										//Output image size:
		{
			++ar;
			fboSize = atoi(argv[ar]);
		}
		else if (opt=="-e")										//Use only trail endpoints:
		{
			only_endpoints = true;
		}
		else if (opt=="-n")										//Use only max so-many-edges from input:
		{
			++ar;
			max_edges = atoi(argv[ar]);
		}
    	else if (opt=="-m")
    	{
     		++ar;
      		mapfile = argv[ar];
    	}
		else if (opt=="-bm") {
			++ar;
			basemapfile = argv[ar];
		}
		else if (opt=="-s") {
			++ar;
			savefile = true;
		}

	}

    glutInitWindowSize(fboSize, fboSize);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
    glutInit(&argc, argv);
	int mainWin = glutCreateWindow("Graph bundling");

    glewInit();                                                 //Must initialize GLEW if we want to use CUDA-OpenGL interop, apparently
    cudaGLSetGLDevice(0);


	gdrawing_bund  = new GraphDrawing();						//Make two graphs: the original one, and the bundled one
	gdrawing_orig  = new GraphDrawing();
	gdrawing_final = new GraphDrawing();

	bool ok = gdrawing_orig->readTrails(graphfile,only_endpoints,max_edges);	//Read some input graph
	if (!ok)
	{
		printf("Error: cannot open graph file '%s'\n", graphfile);
		exit(1);
	}

  	ok = gdrawing_orig->readBackgroundMap(mapfile, fboSize, border); // read shapefile background and points will fit to it
	// ok = gdrawing_orig->readBackgroundMap(mapfile, basemapfile, fboSize, border);

	if (!ok && false)
	{
		printf("Error: cannot open map file '%s'\n", mapfile);
		exit(1);
	}

	bund = new CPUBundling(fboSize);							//Create bundling engine; we'll use it for several graph bundling tasks in this class
	bund->block_endpoints = block_endpoints;
	bund->polyline_style = polyline_style;
	bund->tangent = tangent;
	bund->density_estimation = (CPUBundling::DENSITY_ESTIM)density_estimation;

	gdrawing_orig->normalize(Point2d(fboSize,fboSize), border);		//Fit graph nicely in the graphics window
	gdrawing_orig->draw_points	   = show_points;
	gdrawing_orig->draw_edges      = show_edges;
	gdrawing_orig->draw_endpoints  = show_endpoints;
	gdrawing_orig->color_mode	   = (GraphDrawing::COLOR_MODE)color_mode;
	gdrawing_orig->alpha_mode	   = (GraphDrawing::ALPHA_MODE)alpha_mode;
	gdrawing_orig->densityMap      = bund->h_densityMap;
	gdrawing_orig->shadingMap      = bund->h_shadingMap;
	gdrawing_orig->densityMapSize  = fboSize;
	gdrawing_orig->densityMax	   = &bund->densityMax;
	gdrawing_orig->use_density_alpha = use_density_alpha;
	gdrawing_orig->shading		   = shading;

    *gdrawing_bund  = *gdrawing_orig;                           //Init the bundled graph and the final (drawing) graph with the original graph.

    *gdrawing_final = *gdrawing_bund;                           //This ensures that all drawing options will work, even if we don't bundle anything next.

	if (savefile) saveFile("r");

    glutPostRedisplay();
    glutDisplayFunc(display_cb);

	buildGUI(mainWin);

	glutMainLoop();

	delete gdrawing_bund;
	delete gdrawing_orig;
	delete gdrawing_final;
    return 0;	

}


void display_cb()
{
    printf("Display callback\n");
    postprocess();															//Postprocess the bundling before display
    glClearColor(1,1,1,1);											//Reset main GL state to defaults
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);

		float scales = 1;
		float xscale = 1;
		float yscale = 1;
		float xTrans = 0;
		float yTrans = 0;

		xscale = (float) width/fboSize;
		yscale = (float) height/fboSize;
		scales = min(xscale, yscale); // Get mininmum scale between height and width

    // Get map coordinates and scale given to FboSize viewport
		float xCoord = gdrawing_orig->background_size.x;
		float yCoord = gdrawing_orig->background_size.y;
		float gScale = gdrawing_orig->scale;

    // Calculate current image map boundaries
		float xCurrent = xCoord*gScale*scales;
		float yCurrent = yCoord*gScale*scales;

    // Calculate a new scale to fit map over the entire window if have space
		float new_xscale = width*scales/xCurrent;
		float new_yscale = height*scales/yCurrent;
		scales = min(new_xscale, new_yscale);

    // Set viewport to draw in the entire window size
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);									//Setup projection matrix

    glLoadIdentity();
    // Scale from fboSize to the new scale that fits the window size
    glScalef(scales, scales,1);
    glScalef(scale, scale, scale);

    // Set a plan to the size of the window and not the fboSize
    gluOrtho2D(0, width,0,height);

    // Translate the image from the fboSize to the window as its already scaled
    //to fit the window size
    glTranslatef((width-fboSize)/2, (height-fboSize)/2, 0);
    glTranslatef( obj_pos[0], obj_pos[1], -obj_pos[2] );

    glMatrixMode(GL_MODELVIEW);										//Setup modelview matrix
    glLoadIdentity();

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    TIME(gdrawing_final->draw());											//Draw the final graph

    glutSwapBuffers();												//All done
}


void bundle(int force_update)                                       //Do a full new bundling of gdrawing_orig, save result into gdrawing_bund
{                                                                   //At the end of this, a bundled graph must be available in gdrawing_bund

    if (!force_update && !auto_update) return;                      //If auto_update is off, and we don't force bundling, nothing to do

    *gdrawing_bund = *gdrawing_orig;								//Copy the original drawing to gdrawing_bund, since we want to bundle it,
																	//but we don't want to alter the original drawing.

    bund->setInput(gdrawing_bund);									//Bundle the graph drawing

	if (gpu_bundling)												//Use the CPU or GPU method
		bund->bundleGPU();
	else
		bund->bundleCPU();
}

void postprocess()													//Postprocess the bundling before display
{
        cout<<"P start"<<endl;


	*gdrawing_final = *gdrawing_bund;								//Don't modify the bundled graph, copy it (because we want to redo postprocessing w/o redoing bundling)


	cout<<"Copy"<<endl;

	gdrawing_final->interpolate(*gdrawing_orig,relaxation,dir_separation,max_displacement,!displ_rel_edgelength);
																	//Relax bundling towards original graph, separate edge-directions,
                                                                    //and apply clamping of max-displacement (limit bundling)

        cout<<"Interp"<<endl;


    bool needs_density = use_density_alpha || color_mode==GraphDrawing::DENSITY_MAP;

	if (shading || needs_density)									//Compute density+optionally shading of relaxed graph
		bund->computeDensityShading(gdrawing_final,shading_radius,shading,shading_tube);

	cout<<"shading, P end"<<endl;

}



void control_cb(int ctrl)
{
	switch(ctrl)
	{
	case UI_BUNDLE_ITERATIONS:
	case UI_BUNDLE_MS_ITERATIONS:
	case UI_BUNDLE_KERNEL:
	case UI_BUNDLE_MS_KERNEL:
	case UI_BUNDLE_EDGERES:
	case UI_BUNDLE_SMOOTH:
	case UI_BUNDLE_SMOOTH_ITER:
	case UI_BUNDLE_SMOOTH_ENDS:
	case UI_BUNDLE_SPEED:
	case UI_BUNDLE_CPU_GPU:
	case UI_BUNDLE_DIR_REPULSION:
		bundle();
		break;
	case UI_BUNDLE_DENS_ESTIM:
		bund->density_estimation = (CPUBundling::DENSITY_ESTIM)density_estimation;
		bundle();
		break;
	case UI_BUNDLE_SHAPE:
		bund->initEdgeProfile((CPUBundling::EDGE_PROFILE)bundle_shape);
		bundle();
		break;
	case UI_BUNDLE_BLOCK_ENDS:
		bund->block_endpoints = block_endpoints;
		bund->initEdgeProfile((CPUBundling::EDGE_PROFILE)bundle_shape);
		bundle();
		break;
	case UI_BUNDLE_POLYLINE_STYLE:
		bund->polyline_style = polyline_style;
		bundle();
		break;
	case UI_BUNDLE_DIRECTIONAL:
		bund->tangent = tangent;
		bundle();
		break;
	case UI_BUNDLE_COLOR_MODE:
		gdrawing_orig->color_mode = gdrawing_bund->color_mode = (GraphDrawing::COLOR_MODE)color_mode;
		break;
	case UI_BUNDLE_ALPHA_MODE:
		gdrawing_orig->alpha_mode = gdrawing_bund->alpha_mode = (GraphDrawing::ALPHA_MODE)alpha_mode;
		break;
	case UI_BUNDLE_SHOW_POINTS:
		gdrawing_orig->draw_points = gdrawing_bund->draw_points = show_points;
		break;
	case UI_BUNDLE_SHOW_EDGES:
		gdrawing_orig->draw_edges = gdrawing_bund->draw_edges = show_edges;
		break;
	case UI_BUNDLE_SHOW_ENDPOINTS:
		gdrawing_orig->draw_endpoints = gdrawing_bund->draw_endpoints = show_endpoints;
		break;
  case UI_SHOW_TRANSPORT:
    memcpy(gdrawing_bund->show_transport_types, show_transport_types, sizeof(show_transport_types));
    memcpy(gdrawing_orig->show_transport_types, show_transport_types, sizeof(show_transport_types));
    break;
  case UI_TIME:
		gdrawing_orig->start_time = gdrawing_bund->start_time = float(departureHour + (float)departureMinutes/100.0);
		gdrawing_orig->end_time = gdrawing_bund->end_time = float(arrivalHour + (float)arrivalMinutes/100.0);
    break;
	case UI_BUNDLE_DENSITY_ALPHA:
		gdrawing_orig->use_density_alpha = gdrawing_bund->use_density_alpha = use_density_alpha;
		break;
	case UI_BUNDLE_USE_SHADING:
		gdrawing_orig->shading = gdrawing_bund->shading = shading;
		break;
	case UI_BUNDLE_GLOBAL_ALPHA:
		gdrawing_bund->global_alpha = gdrawing_orig->global_alpha;
		break;
	case UI_BUNDLE_LINEWIDTH:
		gdrawing_bund->line_width = gdrawing_orig->line_width;
		break;
	case UI_BUNDLE_SHADING_AMBIENT:
		gdrawing_bund->amb_factor = gdrawing_orig->amb_factor;
		break;
	case UI_BUNDLE_SHADING_DIFFUSE:
		gdrawing_bund->diff_factor = gdrawing_orig->diff_factor;
		break;
	case UI_BUNDLE_SHADING_SPECULAR:
		gdrawing_bund->spec_factor = gdrawing_orig->spec_factor;
		break;
	case UI_BUNDLE_SHADING_SPECULAR_SIZE:
		gdrawing_bund->spec_highlight_size = gdrawing_orig->spec_highlight_size;
		break;
	case UI_BUNDLE_BACK_IMAGE:
		gdrawing_bund->draw_background = gdrawing_orig->draw_background = draw_background;
		break;
	case UI_BUNDLE_RAILS_IMAGE:
		gdrawing_bund->draw_rails = gdrawing_orig->draw_rails = draw_rails;
		break;
	case UI_BUNDLE_BASEMAP_IMAGE:
		gdrawing_bund->draw_png = gdrawing_orig->draw_png = draw_png;
    case UI_BUNDLE_AUTO_UPDATE:
        if (auto_update) bundle();
        break;


	case UI_BUNDLE_DIR_SEPARATION:
	case UI_BUNDLE_SHADING_RADIUS:
	case UI_BUNDLE_SHADING_TUBE:
	case UI_BUNDLE_RELAXATION:
    case UI_BUNDLE_CLAMP:
    case UI_BUNDLE_CLAMP_ABS_REL:
        break;
    case UI_SAMPLE:
        gdrawing_bund->resample(bund->spl);
        break;
    case UI_BUNDLE:
        bundle(1);
        break;
    case UI_RESET:
        *gdrawing_bund = *gdrawing_orig;
        break;
        case UI_COPY_DRAWING_TO_ORIG:
        *gdrawing_orig = *gdrawing_final;
        *gdrawing_bund = *gdrawing_final;
        break;
    case UI_SCREENSHOT:
        save_screenshot();
        break;
    case UI_SAVE:
        gdrawing_final->saveTrails(save_filename,true);
        cout<<"Bundled data saved to file: "<<save_filename<<endl;
		if (saveFile("w")) cout<<"Configs saved to file: "<< "savefile.dat"<< endl;
        break;
	case UI_QUIT:
		exit(0);
		break;
	}

	glui->post_update_main_gfx();											//Post a redisplay
}


void buildGUI(int mainWin)
{
    GLUI_Control *o1, *o2, *o3, *o4, *o5, *o6, *o7, *o8, *o9, *o10, *o11;
    GLUI_Panel *pan,*pan2,*pan3;											//Construct GUI:
	GLUI_Scrollbar* scr;
	glui = GLUI_Master.create_glui("CUDA Bundling");

	GLUI_Rollout* ui_bundling = glui->add_rollout("Bundling",true);		//4. Panel "Bundling":

	pan3 = glui->add_panel_to_panel(ui_bundling,"Main bundling");
	pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);
	o1 = new GLUI_StaticText(pan,"Iterations");
    o2 = new GLUI_StaticText(pan,"Kernel size");
    o3 = new GLUI_StaticText(pan,"Smoothing factor");
    o4 = new GLUI_StaticText(pan,"Smoothing iterations");
    glui->add_column_to_panel(pan,false);

    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->niter,UI_BUNDLE_ITERATIONS,control_cb);
	scr->set_int_limits(0,40);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->h,UI_BUNDLE_KERNEL,control_cb);
	scr->set_float_limits(3,40);
    o2->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->lambda,UI_BUNDLE_SMOOTH,control_cb);
	scr->set_float_limits(0,1);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->liter,UI_BUNDLE_SMOOTH_ITER,control_cb);
	scr->set_int_limits(0,10);
    o4->set_h(scr->h);

	pan3 = glui->add_panel_to_panel(ui_bundling,"Ends bundling");

    pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);
    o1 = new GLUI_StaticText(pan,"Block endpoints");
    o2 = new GLUI_StaticText(pan,"Iterations");
    o3 = new GLUI_StaticText(pan,"Kernel size");
    o4 = new GLUI_StaticText(pan,"Smoothing factor");
    glui->add_column_to_panel(pan,false);

    new GLUI_Checkbox(pan,"",&block_endpoints,UI_BUNDLE_BLOCK_ENDS,control_cb);
    //o1->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->niter_ms,UI_BUNDLE_MS_ITERATIONS,control_cb);
	scr->set_int_limits(0,40);
    o2->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->h_ms,UI_BUNDLE_MS_KERNEL,control_cb);
	scr->set_float_limits(3,80);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->lambda_ends,UI_BUNDLE_SMOOTH_ENDS,control_cb);
	scr->set_float_limits(0,1);
    o4->set_h(scr->h);

	pan2 = glui->add_panel_to_panel(ui_bundling,"General options");

	pan = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
	o1  = new GLUI_StaticText(pan,"Edge sampling");
    o2  = new GLUI_StaticText(pan,"Advection speed");
    o3  = new GLUI_StaticText(pan,"Dir. bunding repulsion");
    o4  = new GLUI_StaticText(pan,"Relaxation");
    o5  = new GLUI_StaticText(pan,"Max bundle");
    o6  = new GLUI_StaticText(pan,"Max rel. to edgelength");
    o7  = new GLUI_StaticText(pan,"Polyline style");
    o8  = new GLUI_StaticText(pan,"Directional bundling");
    o9  = new GLUI_StaticText(pan,"GPU method");
    o10 = new GLUI_StaticText(pan,"Auto update");

    glui->add_column_to_panel(pan,false);

    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->spl,UI_BUNDLE_EDGERES,control_cb);
	scr->set_float_limits(3,50);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->eps,UI_BUNDLE_SPEED,control_cb);
	scr->set_float_limits(0,1);
    o2->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&bund->rep_strength,UI_BUNDLE_DIR_REPULSION,control_cb);
    scr->set_float_limits(0,1);
    o3->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&relaxation,UI_BUNDLE_RELAXATION,control_cb);
    scr->set_float_limits(0,1);
    o4->set_h(scr->h);
    scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&max_displacement,UI_BUNDLE_CLAMP,control_cb);
    scr->set_float_limits(1.0f/1024,1);
    o5->set_h(scr->h);
    new GLUI_Checkbox(pan,"",&displ_rel_edgelength,UI_BUNDLE_CLAMP_ABS_REL,control_cb);
    //..o6->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"",&polyline_style,UI_BUNDLE_POLYLINE_STYLE,control_cb);
    //..o7->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"",&tangent,UI_BUNDLE_DIRECTIONAL,control_cb);
    //..o8->set_h(checkbox->h)
    new GLUI_Checkbox(pan,"", &gpu_bundling,UI_BUNDLE_CPU_GPU,control_cb);
    new GLUI_Checkbox(pan,"", &auto_update,UI_BUNDLE_AUTO_UPDATE,control_cb);


    pan3 = glui->add_panel_to_panel(pan2,"",GLUI_PANEL_NONE);
        pan = glui->add_panel_to_panel(pan3,"Density estimation");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_dens_estim = new GLUI_RadioGroup(pan,&density_estimation,UI_BUNDLE_DENS_ESTIM,control_cb);
        new GLUI_RadioButton(ui_dens_estim,"Exact");
        new GLUI_RadioButton(ui_dens_estim,"Fast");

        pan = glui->add_panel_to_panel(pan3,"Bundle shape");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_bundle_shape = new GLUI_RadioGroup(pan,&bundle_shape,UI_BUNDLE_SHAPE,control_cb);
        new GLUI_RadioButton(ui_bundle_shape,"FDEB");
        new GLUI_RadioButton(ui_bundle_shape,"HEB");
    //pan3 ready ----------


    pan = glui->add_panel("",GLUI_PANEL_NONE);                              //5. Buttons bar:
    new GLUI_Button(pan,"Resample",UI_SAMPLE,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Bundle",UI_BUNDLE,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Reset",UI_RESET,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Quit",UI_QUIT,control_cb);
    glui->add_column_to_panel(pan,false);
    pan = glui->add_panel("",GLUI_PANEL_NONE);                              //5. Buttons bar:
    new GLUI_Button(pan,"Result->input",UI_COPY_DRAWING_TO_ORIG,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Save",UI_SAVE,control_cb);
    glui->add_column_to_panel(pan,false);
    new GLUI_Button(pan,"Screenshot",UI_SCREENSHOT,control_cb);
    glui->add_column_to_panel(pan,false);


	glui->add_column(true);													//--------------------------------------------------------
	GLUI_Rollout* ui_drawing = glui->add_rollout("Drawing",true);			//6. Roll-out "Drawing":
	pan = glui->add_panel_to_panel(ui_drawing,"Draw what");
	new GLUI_Checkbox(pan,"Edges", &show_edges,UI_BUNDLE_SHOW_EDGES,control_cb);
	new GLUI_Checkbox(pan,"Control points", &show_points,UI_BUNDLE_SHOW_POINTS,control_cb);
	new GLUI_Checkbox(pan,"End points", &show_endpoints,UI_BUNDLE_SHOW_ENDPOINTS,control_cb);
    new GLUI_Checkbox(pan,"Background image",&draw_background,UI_BUNDLE_BACK_IMAGE,control_cb);
    new GLUI_Checkbox(pan,"Rails Lines",&draw_rails,UI_BUNDLE_RAILS_IMAGE,control_cb);
    new GLUI_Checkbox(pan,"Basemap image",&draw_png, UI_BUNDLE_BASEMAP_IMAGE, control_cb);


	pan = glui->add_panel_to_panel(ui_drawing,"",GLUI_PANEL_NONE);
	o1  = new GLUI_StaticText(pan,"Line width");
    o2  = new GLUI_StaticText(pan,"Global alpha");
    o3  = new GLUI_StaticText(pan,"Shading smoothing");
    o4  = new GLUI_StaticText(pan,"Shading ambient");
    o5  = new GLUI_StaticText(pan,"Shading diffuse");
    o6  = new GLUI_StaticText(pan,"Shading specular");
    o7  = new GLUI_StaticText(pan,"Shading highlight");
    o8  = new GLUI_StaticText(pan,"Density-modulated alpha");
    o9  = new GLUI_StaticText(pan,"Illumination");
    o10 = new GLUI_StaticText(pan,"Tube-style shading");
    o11 = new GLUI_StaticText(pan,"Direction separation");

    glui->add_column_to_panel(pan,false);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->line_width,UI_BUNDLE_LINEWIDTH,control_cb);
	scr->set_float_limits(1,5);
    o1->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->global_alpha,UI_BUNDLE_GLOBAL_ALPHA,control_cb);
	scr->set_float_limits(0,1);
    o2->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&shading_radius,UI_BUNDLE_SHADING_RADIUS,control_cb);
	scr->set_float_limits(2,15);
    o3->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->amb_factor,UI_BUNDLE_SHADING_AMBIENT,control_cb);
	scr->set_float_limits(0,1);
    o4->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->diff_factor,UI_BUNDLE_SHADING_DIFFUSE,control_cb);
	scr->set_float_limits(0,1);
    o5->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->spec_factor,UI_BUNDLE_SHADING_SPECULAR,control_cb);
	scr->set_float_limits(0,20);
    o6->set_h(scr->h);
	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&gdrawing_orig->spec_highlight_size,UI_BUNDLE_SHADING_SPECULAR_SIZE,control_cb);
	scr->set_float_limits(1,20);
    o7->set_h(scr->h);

	new GLUI_Checkbox(pan,"",&use_density_alpha,UI_BUNDLE_DENSITY_ALPHA,control_cb);
	new GLUI_Checkbox(pan,"",&shading,UI_BUNDLE_USE_SHADING,control_cb);
	new GLUI_Checkbox(pan,"",&shading_tube,UI_BUNDLE_SHADING_TUBE,control_cb);

	scr = new GLUI_Scrollbar(pan,"",GLUI_SCROLL_HORIZONTAL,&dir_separation,UI_BUNDLE_DIR_SEPARATION,control_cb);
	scr->set_float_limits(-20,20);
    o11->set_h(scr->h);



    pan3 = glui->add_panel_to_panel(ui_drawing,"",GLUI_PANEL_NONE);

        pan = glui->add_panel_to_panel(pan3,"Coloring");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_color = new GLUI_RadioGroup(pan,&color_mode,UI_BUNDLE_COLOR_MODE,control_cb);
        new GLUI_RadioButton(ui_color,"Grayscale (length)");
        new GLUI_RadioButton(ui_color,"Blue-red (length)");
        new GLUI_RadioButton(ui_color,"Red-blue (length)");
        new GLUI_RadioButton(ui_color,"Directional");
        new GLUI_RadioButton(ui_color,"Black");
        new GLUI_RadioButton(ui_color,"Density map");
        new GLUI_RadioButton(ui_color,"Displacement");
        new GLUI_RadioButton(ui_color,"Transport Types");
        new GLUI_RadioButton(ui_color,"Transport Public x Private");
        new GLUI_RadioButton(ui_color,"Transport Motorized x Non Motorized");

        pan = glui->add_panel_to_panel(pan3,"Transparency");
        glui->add_column_to_panel(pan3,false);
        GLUI_RadioGroup* ui_alpha = new GLUI_RadioGroup(pan,&alpha_mode,UI_BUNDLE_ALPHA_MODE,control_cb);
        new GLUI_RadioButton(ui_alpha,"Constant");
        new GLUI_RadioButton(ui_alpha,"Mark short edges");
        new GLUI_RadioButton(ui_alpha,"Mark long edges");


    pan3 = glui->add_panel_to_panel(ui_drawing,"View Control",GLUI_PANEL_EMBOSSED);

        glui->add_column_to_panel(pan3,false);
        pan = glui->add_panel_to_panel(pan3,"", GLUI_PANEL_NONE);
        GLUI_Translation* ui_translation = new GLUI_Translation(pan3,"Translation", GLUI_TRANSLATION_XY, obj_pos);
        ui_translation->set_w(1);

        pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);
        GLUI_Spinner *scale_spinner = new GLUI_Spinner(pan, "Scale", &scale);
        scale_spinner->set_float_limits( .2f, 8.0 ); //Magic number for scaling, must be proportional to map scale


    pan3 = glui->add_panel_to_panel(ui_drawing,"Time Control",GLUI_PANEL_EMBOSSED);
        glui->add_column_to_panel(pan3,false);
        pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);

        glui->add_column_to_panel(pan,false);
        GLUI_Spinner *time_spinner = new GLUI_Spinner(pan, "Departure H", &departureHour, UI_TIME, control_cb);
        time_spinner->set_int_limits( 0, 23 );

        glui->add_column_to_panel(pan,false);
        time_spinner = new GLUI_Spinner(pan, "min", &departureMinutes, UI_TIME, control_cb);
        time_spinner->set_int_limits( 0, 59 );

        pan = glui->add_panel_to_panel(pan3,"",GLUI_PANEL_NONE);

        glui->add_column_to_panel(pan3,false);
        time_spinner = new GLUI_Spinner(pan, "Arrival: H", &arrivalHour, UI_TIME, control_cb);
        time_spinner->set_int_limits( 0, 23 );

        glui->add_column_to_panel(pan,false);
        time_spinner = new GLUI_Spinner(pan, "min", &arrivalMinutes, UI_TIME, control_cb);
        time_spinner->set_int_limits( 0, 59 );

	glui->add_column(true);													//--------------------------------------------------------
	ui_drawing = glui->add_rollout("Transportation",true);			//6. Roll-out "Drawing":
	pan = glui->add_panel_to_panel(ui_drawing,"Draw what");
	new GLUI_Checkbox(pan,"Metro", &show_transport_types[METRO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Train", &show_transport_types[TREM],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Monorail", &show_transport_types[MONOTRILHO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Bus from SP", &show_transport_types[ONIBUS_DE_SAO_PAULO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Bus from other regions", &show_transport_types[ONIBUS_OUTROS_MUNICIPIOS],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Metropolitan Bus", &show_transport_types[ONIBUS_METROPOLITANO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Chartered Transport", &show_transport_types[TRANSPORTE_FRETADO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"School Bus", &show_transport_types[TRANSPORTE_ESCOLAR],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Driving Car", &show_transport_types[DIRIGINDO_AUTOMOVEL],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Car Passenger", &show_transport_types[PASSAGEIRO_DE_AUTOMOVEL],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Regular Taxi", &show_transport_types[TAXI_CONVENCIONAL],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Non Regular Taxi", &show_transport_types[TAXI_NAO_CONVENCIONAL],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Driving Motorcycle", &show_transport_types[DIRIGINDO_MOTO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Motorcycle Passenger", &show_transport_types[PASSAGEIRO_DE_MOTO],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Bicycle", &show_transport_types[BICICLETA],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"On foot", &show_transport_types[A_PE],UI_SHOW_TRANSPORT,control_cb);
	new GLUI_Checkbox(pan,"Others", &show_transport_types[OUTROS],UI_SHOW_TRANSPORT,control_cb);

	glui->set_main_gfx_window(mainWin);										//Link GLUI with GLUT (seems needed)
}

int saveFile(const char *mode) {
	FILE *fp = fopen("savefile.dat", mode);

	if (!fp) {
		printf("saveFile Error: Erro ao abrir o arquivo de configurações.\n");
		return false;
	}

	if (strcmp("r", mode) == 0) {
		if (fread(&bund->niter, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&bund->h, sizeof(float), 1, fp) != 1) return false; 
    	if (fread(&bund->lambda, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&bund->liter, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&block_endpoints, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&bund->niter_ms, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&bund->h_ms, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&bund->lambda_ends, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&bund->spl, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&bund->eps, sizeof(float), 1, fp) != 1) return false; 
    	if (fread(&bund->rep_strength, sizeof(float), 1, fp) != 1) return false; 
    	if (fread(&relaxation, sizeof(float), 1, fp) != 1) return false; 
    	if (fread(&max_displacement, sizeof(float), 1, fp) != 1) return false; 
    	if (fread(&displ_rel_edgelength, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&polyline_style, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&tangent, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&gpu_bundling, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&auto_update, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&density_estimation, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&bundle_shape, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&show_edges, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&show_points, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&show_endpoints, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&draw_background, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&draw_rails, sizeof(int), 1, fp) != 1) return false; 
    	if (fread(&draw_png, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->line_width, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->global_alpha, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&shading_radius, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->amb_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->diff_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->spec_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&gdrawing_orig->spec_highlight_size, sizeof(float), 1, fp) != 1) return false; 
		if (fread(&use_density_alpha, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&shading, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&shading_tube, sizeof(int), 1, fp) != 1) return false; 
		if (fread(&dir_separation, sizeof(float), 1, fp) != 1) return false; 	
	
	} else if(strcmp("w", mode) == 0) {
		if (fwrite(&bund->niter, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&bund->h, sizeof(float), 1, fp) != 1) return false; 
    	if (fwrite(&bund->lambda, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&bund->liter, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&block_endpoints, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&bund->niter_ms, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&bund->h_ms, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&bund->lambda_ends, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&bund->spl, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&bund->eps, sizeof(float), 1, fp) != 1) return false; 
    	if (fwrite(&bund->rep_strength, sizeof(float), 1, fp) != 1) return false; 
    	if (fwrite(&relaxation, sizeof(float), 1, fp) != 1) return false; 
    	if (fwrite(&max_displacement, sizeof(float), 1, fp) != 1) return false; 
    	if (fwrite(&displ_rel_edgelength, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&polyline_style, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&tangent, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&gpu_bundling, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&auto_update, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&density_estimation, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&bundle_shape, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&show_edges, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&show_points, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&show_endpoints, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&draw_background, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&draw_rails, sizeof(int), 1, fp) != 1) return false; 
    	if (fwrite(&draw_png, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->line_width, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->global_alpha, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&shading_radius, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->amb_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->diff_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->spec_factor, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&gdrawing_orig->spec_highlight_size, sizeof(float), 1, fp) != 1) return false; 
		if (fwrite(&use_density_alpha, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&shading, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&shading_tube, sizeof(int), 1, fp) != 1) return false; 
		if (fwrite(&dir_separation, sizeof(float), 1, fp) != 1) return false; 	
	
	} else {
		printf("saveFile Error: Operação não suportada.\n");
		return false;
	}



	fclose(fp);
	return true;
}