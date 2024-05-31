/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include "FloodPopulation.h"
#include "OBJModel.h"
#include "BufferObjects.h"

/** Macro for toggling drawing of the navigation map wireframe grid */
int drawGrid2 = 0;

/** Macro for toggling drawing of the navigation map arrows */
int drawArrows2  = 1;

/** Macro for toggling the use of a single large vbo */
BOOLEAN useLargeVBO2 = TRUE;

//floodcell map width
int fc_width;

//floodcell instances
GLuint fc_instances_tbo;
GLuint fc_instances_tex;
cudaGraphicsResource_t fc_instances_cgr;

//model primative counts
int arrow_v_count2;
int arrow_f_count2;
//model primative data
glm::vec3* arrow_vertices2;
glm::vec3* arrow_normals2;
glm::ivec3* arrow_faces2;
//model buffer obejcts
GLuint arrow_verts_vbo2;
GLuint arrow_elems_vbo2;

//vertex attribute buffer (for single large vbo)
GLuint arrow_attributes_vbo2;

//Shader and shader attribute pointers
GLuint fc_vertexShader;
GLuint fc_shaderProgram;
GLuint fcvs_instance_map;
GLuint fcvs_instance_index;
GLuint fcvs_FC_WIDTH;
GLuint fcvs_ENV_MAX;
GLuint fcvs_ENV_WIDTH;

//external prototypes imported from FLAME GPU
extern int get_agent_flood_MAX_count();
extern int get_agent_flood_static_count();

//PRIVATE PROTOTYPES
/** createfloodBufferObjects
 * Creates all Buffer Objects for instancing and model data
 */
void createfloodBufferObjects();
/** initfloodShader
 * Initialises the Flood Cell Shader and shader attributes
 */
void initfloodShader();


void initfloodPopulation()
{
	float scale;

	fc_width = (int)floor(sqrt((float)get_agent_flood_MAX_count()));

	printf("fc_width is: %d \n", fc_width);
	printf("flood_MAX_count is: %f \n", (float)get_agent_flood_MAX_count());

	arrow_v_count2 = 25;
	arrow_f_count2 = 46;

	//load cone model
	allocateObjModel(arrow_v_count2, arrow_f_count2, &arrow_vertices2, &arrow_normals2, &arrow_faces2);
	loadObjFromFile("../../media/cone_Copy_2.obj",	arrow_v_count2, arrow_f_count2, arrow_vertices2, arrow_normals2, arrow_faces2);
	scale = ENV_MAX/(float)fc_width;
	scaleObj(scale, arrow_v_count2, arrow_vertices2);		 
	
	printf("scale of flood is %f \n", scale);

	createfloodBufferObjects();
	initfloodShader();
	displayMapNumber2(0);
}


void renderfloodPopulation()
{	
	int i, x, y;

	if (drawArrows2)
	{
		//generate instance data from FLAME GPU model
		generate_instances2(&fc_instances_tbo, &fc_instances_cgr);

		
		//bind vertex program
		glUseProgram(fc_shaderProgram);

		//bind instance data
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, fc_instances_tex);
		glUniform1i(fcvs_instance_map, 0);


		if (useLargeVBO2)
		{
			glBindBuffer(GL_ARRAY_BUFFER, arrow_attributes_vbo2);
			glEnableVertexAttribArray(fcvs_instance_index);
			glVertexAttribPointer(fcvs_instance_index, 1, GL_FLOAT, 0, 0, 0);

			glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo2);
			glVertexPointer(3, GL_FLOAT, 0, 0);
			
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo2);

			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
		    
			glDrawElements(GL_TRIANGLES, arrow_f_count2*3*get_agent_flood_static_count(), GL_UNSIGNED_INT, 0);

			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);
			glDisableVertexAttribArray(fcvs_instance_index);
		}
		else
		{
			//draw arrows
			for (i=0; i<get_agent_flood_static_count(); i++)
			{
				glVertexAttrib1f(fcvs_instance_index, (float)i);
				
				glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo2);
				glVertexPointer(3, GL_FLOAT, 0, 0);

				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo2);

				glEnableClientState(GL_VERTEX_ARRAY);
				glEnableClientState(GL_ELEMENT_ARRAY_BUFFER);
			    
				glDrawElements(GL_TRIANGLES, arrow_f_count2*3, GL_UNSIGNED_INT, 0);

				glDisableClientState(GL_VERTEX_ARRAY);
				glDisableClientState(GL_ELEMENT_ARRAY_BUFFER);

			}
		}

		glUseProgram(0);
	}

	if (drawGrid2)
	{
		//draw line grid
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glBegin(GL_QUADS);
		{
			for (y=0; y<fc_width; y++){
				for (x=0; x<fc_width; x++){
					float x_min = (float)(x)/((float)fc_width/(float)ENV_WIDTH)-ENV_MAX;
					float x_max = (float)(x+1)/((float)fc_width/(float)ENV_WIDTH)-ENV_MAX;
					float y_min = (float)(y)/((float)fc_width/(float)ENV_WIDTH)-ENV_MAX;
					float y_max = (float)(y+1)/((float)fc_width/(float)ENV_WIDTH)-ENV_MAX;

					glVertex2f(x_min, y_min);
					glVertex2f(x_min, y_max);
					glVertex2f(x_max, y_max);
					glVertex2f(x_max, y_min);
				}
			}
		}
		glEnd();
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
}


void createfloodBufferObjects()
{
	//create TBO
	createTBO(&fc_instances_tbo, &fc_instances_tex, get_agent_flood_MAX_count()* sizeof(glm::vec4));
	registerBO(&fc_instances_cgr, &fc_instances_tbo);

	if (useLargeVBO2)
	{
		int i,v,f = 0;
		glm::vec3* verts;
		glm::ivec3* faces;
		float* atts;

		//create VBOs
		createVBO(&arrow_verts_vbo2, GL_ARRAY_BUFFER, get_agent_flood_MAX_count()*arrow_v_count2*sizeof(glm::vec3));
		createVBO(&arrow_elems_vbo2, GL_ELEMENT_ARRAY_BUFFER, get_agent_flood_MAX_count()*arrow_f_count2*sizeof(glm::ivec3));
		//create attributes vbo
		createVBO(&arrow_attributes_vbo2, GL_ARRAY_BUFFER, get_agent_flood_MAX_count()*arrow_v_count2*sizeof(int));
		
		
		//bind and map vertex data
		glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo2);
		verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_flood_MAX_count();i++){
			int offset = i*arrow_v_count2;
			// int x = floor(i/64.0f);
			// int y = i%64;
			for (v=0;v<arrow_v_count2;v++){
				verts[offset+v][0] = arrow_vertices2[v][0];
				verts[offset+v][1] = arrow_vertices2[v][1];
				verts[offset+v][2] = arrow_vertices2[v][2];
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer( GL_ARRAY_BUFFER, 0);

		//bind and map face data
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo2);
		faces = (glm::ivec3*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_flood_MAX_count();i++){
			int offset = i*arrow_f_count2;
			int vert_offset = i*arrow_v_count2;	//need to offset all face indices by number of verts in each model
			for (f=0;f<arrow_f_count2;f++){
				faces[offset+f][0] = arrow_faces2[f][0]+vert_offset;
				faces[offset+f][1] = arrow_faces2[f][1]+vert_offset;
				faces[offset+f][2] = arrow_faces2[f][2]+vert_offset;
			}
		}
		glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0);

		
		//bind and map vbo attrbiute data
		glBindBuffer(GL_ARRAY_BUFFER, arrow_attributes_vbo2);
		atts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		for (i=0;i<get_agent_flood_MAX_count();i++){
			int offset = i*arrow_v_count2;
			for (v=0;v<arrow_v_count2;v++){
				atts[offset+v] = (float)i;
			}
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
		

		checkGLError();
	}
	else
	{
		//create VBOs
		createVBO(&arrow_verts_vbo2, GL_ARRAY_BUFFER, arrow_v_count2*sizeof(glm::vec3));
		createVBO(&arrow_elems_vbo2, GL_ELEMENT_ARRAY_BUFFER, arrow_f_count2*sizeof(glm::ivec3));

		//bind VBOs
		glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo2);
		glBufferData(GL_ARRAY_BUFFER, arrow_v_count2*sizeof(glm::vec3), arrow_vertices2, GL_DYNAMIC_DRAW);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo2);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, arrow_f_count2*sizeof(glm::ivec3), arrow_faces2, GL_DYNAMIC_DRAW);
		glBindBuffer( GL_ARRAY_BUFFER, 0);
	}
	

}

void initfloodShader()
{
	const char* v = flood_vshader_source;
	int status;

	//vertex shader
	fc_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(fc_vertexShader, 1, &v, 0);
    glCompileShader(fc_vertexShader);

	//program
    fc_shaderProgram = glCreateProgram();
    glAttachShader(fc_shaderProgram, fc_vertexShader);
    glLinkProgram(fc_shaderProgram);

	// check for errors
	glGetShaderiv(fc_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		char data[1024];
		int len;
		printf("ERROR: Shader Compilation Error\n");
		glGetShaderInfoLog(fc_vertexShader, 1024, &len, data); 
		printf("%s", data);
	}
	glGetProgramiv(fc_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}

	// get shader variables
	fcvs_instance_map = glGetUniformLocation(fc_shaderProgram, "instance_map");
	fcvs_instance_index = glGetAttribLocation(fc_shaderProgram, "instance_index"); 
	fcvs_FC_WIDTH = glGetUniformLocation(fc_shaderProgram, "FC_WIDTH");
	fcvs_ENV_MAX = glGetUniformLocation(fc_shaderProgram, "ENV_MAX");
	fcvs_ENV_WIDTH = glGetUniformLocation(fc_shaderProgram, "ENV_WIDTH");

	//set uniforms (need to use prgram to do so)
	glUseProgram(fc_shaderProgram);
	glUniform1f(fcvs_FC_WIDTH, (float)fc_width);
	glUniform1f(fcvs_ENV_MAX, ENV_MAX);
	glUniform1f(fcvs_ENV_WIDTH, ENV_WIDTH);
	glUseProgram(0);
}

void toggleGridDisplayOnOff2()
{
	drawGrid2 = !drawGrid2;
}

void setArrowsDisplayOnOff2(TOGGLE_STATE state)
{
	drawArrows2 = state;
}


void toggleArrowsDisplayOnOff2()
{
	drawArrows2 = !drawArrows2;
}

int getActiveExit2()
{
	return getCurrentMap2();
}
