
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <algorithm>
#include <string>
#include <vector>



#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( int (*parseFunc)(const char*), char* buffer, short *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (short)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: variable array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_household_list* h_households_default, xmachine_memory_household_list* d_households_default, int h_xmachine_memory_household_default_count,xmachine_memory_flood_list* h_floods_static, xmachine_memory_flood_list* d_floods_static, int h_xmachine_memory_flood_static_count,xmachine_memory_warning_list* h_warnings_static_warning, xmachine_memory_warning_list* d_warnings_static_warning, int h_xmachine_memory_warning_static_warning_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_households_default, d_households_default, sizeof(xmachine_memory_household_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying household Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_floods_static, d_floods_static, sizeof(xmachine_memory_flood_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying flood Agent static State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_warnings_static_warning, d_warnings_static_warning, sizeof(xmachine_memory_warning_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying warning Agent static_warning State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
    fputs("\t<FLOOD_DATA_ARRAY>", file);
    for (int j=0;j<900000;j++){
        fprintf(file, "%d", get_FLOOD_DATA_ARRAY()[j]);
        if(j!=(900000-1))
            fprintf(file, ",");
    }
    fputs("</FLOOD_DATA_ARRAY>\n", file);
    fputs("\t<TIME>", file);
    sprintf(data, "%d", (*get_TIME()));
    fputs(data, file);
    fputs("</TIME>\n", file);
    fputs("\t<RANDOM_SEED_SEC>", file);
    sprintf(data, "%d", (*get_RANDOM_SEED_SEC()));
    fputs(data, file);
    fputs("</RANDOM_SEED_SEC>\n", file);
    fputs("\t<RANDOM_SEED_MIN>", file);
    sprintf(data, "%d", (*get_RANDOM_SEED_MIN()));
    fputs(data, file);
    fputs("</RANDOM_SEED_MIN>\n", file);
    fputs("\t<TIME_SCALER>", file);
    sprintf(data, "%f", (*get_TIME_SCALER()));
    fputs(data, file);
    fputs("</TIME_SCALER>\n", file);
	fputs("</environment>\n" , file);

	//Write each household agent to xml
	for (int i=0; i<h_xmachine_memory_household_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>household</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_households_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_households_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<resident_num>", file);
        sprintf(data, "%d", h_households_default->resident_num[i]);
		fputs(data, file);
		fputs("</resident_num>\n", file);
        
		fputs("<OYI>", file);
        sprintf(data, "%d", h_households_default->OYI[i]);
		fputs(data, file);
		fputs("</OYI>\n", file);
        
		fputs("<tenure>", file);
        sprintf(data, "%d", h_households_default->tenure[i]);
		fputs(data, file);
		fputs("</tenure>\n", file);
        
		fputs("<take_measure>", file);
        sprintf(data, "%d", h_households_default->take_measure[i]);
		fputs(data, file);
		fputs("</take_measure>\n", file);
        
		fputs("<warning_area>", file);
        sprintf(data, "%d", h_households_default->warning_area[i]);
		fputs(data, file);
		fputs("</warning_area>\n", file);
        
		fputs("<get_warning>", file);
        sprintf(data, "%d", h_households_default->get_warning[i]);
		fputs(data, file);
		fputs("</get_warning>\n", file);
        
		fputs("<alert_state>", file);
        sprintf(data, "%d", h_households_default->alert_state[i]);
		fputs(data, file);
		fputs("</alert_state>\n", file);
        
		fputs("<sandbag_state>", file);
        sprintf(data, "%d", h_households_default->sandbag_state[i]);
		fputs(data, file);
		fputs("</sandbag_state>\n", file);
        
		fputs("<sandbag_time_count>", file);
        sprintf(data, "%d", h_households_default->sandbag_time_count[i]);
		fputs(data, file);
		fputs("</sandbag_time_count>\n", file);
        
		fputs("<flooded_time>", file);
        sprintf(data, "%d", h_households_default->flooded_time[i]);
		fputs(data, file);
		fputs("</flooded_time>\n", file);
        
		fputs("<initial_wl>", file);
        sprintf(data, "%f", h_households_default->initial_wl[i]);
		fputs(data, file);
		fputs("</initial_wl>\n", file);
        
		fputs("<actual_wl>", file);
        sprintf(data, "%f", h_households_default->actual_wl[i]);
		fputs(data, file);
		fputs("</actual_wl>\n", file);
        
		fputs("<average_wl>", file);
        sprintf(data, "%f", h_households_default->average_wl[i]);
		fputs(data, file);
		fputs("</average_wl>\n", file);
        
		fputs("<max_wl>", file);
        sprintf(data, "%f", h_households_default->max_wl[i]);
		fputs(data, file);
		fputs("</max_wl>\n", file);
        
		fputs("<financial_damage>", file);
        sprintf(data, "%f", h_households_default->financial_damage[i]);
		fputs(data, file);
		fputs("</financial_damage>\n", file);
        
		fputs("<inform_others>", file);
        sprintf(data, "%d", h_households_default->inform_others[i]);
		fputs(data, file);
		fputs("</inform_others>\n", file);
        
		fputs("<get_informed>", file);
        sprintf(data, "%d", h_households_default->get_informed[i]);
		fputs(data, file);
		fputs("</get_informed>\n", file);
        
		fputs("<lod>", file);
        sprintf(data, "%d", h_households_default->lod[i]);
		fputs(data, file);
		fputs("</lod>\n", file);
        
		fputs("<animate>", file);
        sprintf(data, "%f", h_households_default->animate[i]);
		fputs(data, file);
		fputs("</animate>\n", file);
        
		fputs("<animate_dir>", file);
        sprintf(data, "%d", h_households_default->animate_dir[i]);
		fputs(data, file);
		fputs("</animate_dir>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each flood agent to xml
	for (int i=0; i<h_xmachine_memory_flood_static_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>flood</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_floods_static->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_floods_static->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<floodID>", file);
        sprintf(data, "%d", h_floods_static->floodID[i]);
		fputs(data, file);
		fputs("</floodID>\n", file);
        
		fputs("<flood_h>", file);
        sprintf(data, "%f", h_floods_static->flood_h[i]);
		fputs(data, file);
		fputs("</flood_h>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each warning agent to xml
	for (int i=0; i<h_xmachine_memory_warning_static_warning_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>warning</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_warnings_static_warning->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_warnings_static_warning->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<flooded_households>", file);
        sprintf(data, "%d", h_warnings_static_warning->flooded_households[i]);
		fputs(data, file);
		fputs("</flooded_households>\n", file);
        
		fputs("<total_financial_damage>", file);
        sprintf(data, "%f", h_warnings_static_warning->total_financial_damage[i]);
		fputs(data, file);
		fputs("</total_financial_damage>\n", file);
        
		fputs("<total_take_measure>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_take_measure[i]);
		fputs(data, file);
		fputs("</total_take_measure>\n", file);
        
		fputs("<total_get_warning>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_get_warning[i]);
		fputs(data, file);
		fputs("</total_get_warning>\n", file);
        
		fputs("<total_alert_state>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_alert_state[i]);
		fputs(data, file);
		fputs("</total_alert_state>\n", file);
        
		fputs("<total_sandbag1>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_sandbag1[i]);
		fputs(data, file);
		fputs("</total_sandbag1>\n", file);
        
		fputs("<total_sandbag2>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_sandbag2[i]);
		fputs(data, file);
		fputs("</total_sandbag2>\n", file);
        
		fputs("<total_sandbag3>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_sandbag3[i]);
		fputs(data, file);
		fputs("</total_sandbag3>\n", file);
        
		fputs("<total_inform_others>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_inform_others[i]);
		fputs(data, file);
		fputs("</total_inform_others>\n", file);
        
		fputs("<total_get_informed>", file);
        sprintf(data, "%d", h_warnings_static_warning->total_get_informed[i]);
		fputs(data, file);
		fputs("</total_get_informed>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_household_list* h_households, int* h_xmachine_memory_household_count,xmachine_memory_flood_list* h_floods, int* h_xmachine_memory_flood_count,xmachine_memory_warning_list* h_warnings, int* h_xmachine_memory_warning_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;
    int in_household_x;
    int in_household_y;
    int in_household_resident_num;
    int in_household_OYI;
    int in_household_tenure;
    int in_household_take_measure;
    int in_household_warning_area;
    int in_household_get_warning;
    int in_household_alert_state;
    int in_household_sandbag_state;
    int in_household_sandbag_time_count;
    int in_household_flooded_time;
    int in_household_initial_wl;
    int in_household_actual_wl;
    int in_household_average_wl;
    int in_household_max_wl;
    int in_household_financial_damage;
    int in_household_inform_others;
    int in_household_get_informed;
    int in_household_lod;
    int in_household_animate;
    int in_household_animate_dir;
    int in_flood_x;
    int in_flood_y;
    int in_flood_floodID;
    int in_flood_flood_h;
    int in_warning_x;
    int in_warning_y;
    int in_warning_flooded_households;
    int in_warning_total_financial_damage;
    int in_warning_total_take_measure;
    int in_warning_total_get_warning;
    int in_warning_total_alert_state;
    int in_warning_total_sandbag1;
    int in_warning_total_sandbag2;
    int in_warning_total_sandbag3;
    int in_warning_total_inform_others;
    int in_warning_total_get_informed;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_FLOOD_DATA_ARRAY;
    
    int in_env_TIME;
    
    int in_env_RANDOM_SEED_SEC;
    
    int in_env_RANDOM_SEED_MIN;
    
    int in_env_TIME_SCALER;
    
	/* set agent count to zero */
	*h_xmachine_memory_household_count = 0;
	*h_xmachine_memory_flood_count = 0;
	*h_xmachine_memory_warning_count = 0;
	
	/* Variables for initial state data */
	float household_x;
	float household_y;
	int household_resident_num;
	int household_OYI;
	int household_tenure;
	int household_take_measure;
	int household_warning_area;
	int household_get_warning;
	int household_alert_state;
	int household_sandbag_state;
	int household_sandbag_time_count;
	int household_flooded_time;
	float household_initial_wl;
	float household_actual_wl;
	float household_average_wl;
	float household_max_wl;
	float household_financial_damage;
	int household_inform_others;
	int household_get_informed;
	int household_lod;
	float household_animate;
	int household_animate_dir;
	int flood_x;
	int flood_y;
	int flood_floodID;
	float flood_flood_h;
	float warning_x;
	float warning_y;
	int warning_flooded_households;
	float warning_total_financial_damage;
	int warning_total_take_measure;
	int warning_total_get_warning;
	int warning_total_alert_state;
	int warning_total_sandbag1;
	int warning_total_sandbag2;
	int warning_total_sandbag3;
	int warning_total_inform_others;
	int warning_total_get_informed;

    /* Variables for environment variables */
    
    // short env_FLOOD_DATA_ARRAY[900000];
	
	short* env_FLOOD_DATA_ARRAY= new short[900000];

    int env_TIME;
        int env_RANDOM_SEED_SEC;
        int env_RANDOM_SEED_MIN;
        float env_TIME_SCALER;
        

    printf("initial------------\n");

	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_household_x = 0;
	in_household_y = 0;
	in_household_resident_num = 0;
	in_household_OYI = 0;
	in_household_tenure = 0;
	in_household_take_measure = 0;
	in_household_warning_area = 0;
	in_household_get_warning = 0;
	in_household_alert_state = 0;
	in_household_sandbag_state = 0;
	in_household_sandbag_time_count = 0;
	in_household_flooded_time = 0;
	in_household_initial_wl = 0;
	in_household_actual_wl = 0;
	in_household_average_wl = 0;
	in_household_max_wl = 0;
	in_household_financial_damage = 0;
	in_household_inform_others = 0;
	in_household_get_informed = 0;
	in_household_lod = 0;
	in_household_animate = 0;
	in_household_animate_dir = 0;
	in_flood_x = 0;
	in_flood_y = 0;
	in_flood_floodID = 0;
	in_flood_flood_h = 0;
	in_warning_x = 0;
	in_warning_y = 0;
	in_warning_flooded_households = 0;
	in_warning_total_financial_damage = 0;
	in_warning_total_take_measure = 0;
	in_warning_total_get_warning = 0;
	in_warning_total_alert_state = 0;
	in_warning_total_sandbag1 = 0;
	in_warning_total_sandbag2 = 0;
	in_warning_total_sandbag3 = 0;
	in_warning_total_inform_others = 0;
	in_warning_total_get_informed = 0;
    in_env_FLOOD_DATA_ARRAY = 0;
    in_env_TIME = 0;
    in_env_RANDOM_SEED_SEC = 0;
    in_env_RANDOM_SEED_MIN = 0;
    in_env_TIME_SCALER = 0;
	//set all household values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_household_MAX; k++)
	{	
		h_households->x[k] = 0;
		h_households->y[k] = 0;
		h_households->resident_num[k] = 0;
		h_households->OYI[k] = 0;
		h_households->tenure[k] = 0;
		h_households->take_measure[k] = 0;
		h_households->warning_area[k] = 0;
		h_households->get_warning[k] = 0;
		h_households->alert_state[k] = 0;
		h_households->sandbag_state[k] = 0;
		h_households->sandbag_time_count[k] = 0;
		h_households->flooded_time[k] = 0;
		h_households->initial_wl[k] = 0;
		h_households->actual_wl[k] = 0;
		h_households->average_wl[k] = 0;
		h_households->max_wl[k] = 0;
		h_households->financial_damage[k] = 0;
		h_households->inform_others[k] = 0;
		h_households->get_informed[k] = 0;
		h_households->lod[k] = 0;
		h_households->animate[k] = 0;
		h_households->animate_dir[k] = 0;
	}
	
	//set all flood values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_flood_MAX; k++)
	{	
		h_floods->x[k] = 0;
		h_floods->y[k] = 0;
		h_floods->floodID[k] = 0;
		h_floods->flood_h[k] = 0;
	}
	
	//set all warning values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_warning_MAX; k++)
	{	
		h_warnings->x[k] = 0;
		h_warnings->y[k] = 0;
		h_warnings->flooded_households[k] = 0;
		h_warnings->total_financial_damage[k] = 0;
		h_warnings->total_take_measure[k] = 0;
		h_warnings->total_get_warning[k] = 0;
		h_warnings->total_alert_state[k] = 0;
		h_warnings->total_sandbag1[k] = 0;
		h_warnings->total_sandbag2[k] = 0;
		h_warnings->total_sandbag3[k] = 0;
		h_warnings->total_inform_others[k] = 0;
		h_warnings->total_get_informed[k] = 0;
	}
	

	/* Default variables for memory */
    household_x = 0;
    household_y = 0;
    household_resident_num = 0;
    household_OYI = 0;
    household_tenure = 0;
    household_take_measure = 0;
    household_warning_area = 0;
    household_get_warning = 0;
    household_alert_state = 0;
    household_sandbag_state = 0;
    household_sandbag_time_count = 0;
    household_flooded_time = 0;
    household_initial_wl = 0;
    household_actual_wl = 0;
    household_average_wl = 0;
    household_max_wl = 0;
    household_financial_damage = 0;
    household_inform_others = 0;
    household_get_informed = 0;
    household_lod = 0;
    household_animate = 0;
    household_animate_dir = 0;
    flood_x = 0;
    flood_y = 0;
    flood_floodID = 0;
    flood_flood_h = 0;
    warning_x = 0;
    warning_y = 0;
    warning_flooded_households = 0;
    warning_total_financial_damage = 0;
    warning_total_take_measure = 0;
    warning_total_get_warning = 0;
    warning_total_alert_state = 0;
    warning_total_sandbag1 = 0;
    warning_total_sandbag2 = 0;
    warning_total_sandbag3 = 0;
    warning_total_inform_others = 0;
    warning_total_get_informed = 0;

    /* Default variables for environment variables */
    
    for (i=0;i<900000;i++){
        env_FLOOD_DATA_ARRAY[i] = 0;
    }
    env_TIME = 0;
    env_RANDOM_SEED_SEC = 0;
    env_RANDOM_SEED_MIN = 0;
    env_TIME_SCALER = 0;
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a > check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '>' && i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '>' && i >= 2 && buffer[i-1] == '-' && buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "household") == 0)
				{
					if (*h_xmachine_memory_household_count > xmachine_memory_household_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent household exceeded whilst reading data\n", xmachine_memory_household_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_households->x[*h_xmachine_memory_household_count] = household_x;//Check maximum x value
                    if(agent_maximum.x < household_x)
                        agent_maximum.x = (float)household_x;
                    //Check minimum x value
                    if(agent_minimum.x > household_x)
                        agent_minimum.x = (float)household_x;
                    
					h_households->y[*h_xmachine_memory_household_count] = household_y;//Check maximum y value
                    if(agent_maximum.y < household_y)
                        agent_maximum.y = (float)household_y;
                    //Check minimum y value
                    if(agent_minimum.y > household_y)
                        agent_minimum.y = (float)household_y;
                    
					h_households->resident_num[*h_xmachine_memory_household_count] = household_resident_num;
					h_households->OYI[*h_xmachine_memory_household_count] = household_OYI;
					h_households->tenure[*h_xmachine_memory_household_count] = household_tenure;
					h_households->take_measure[*h_xmachine_memory_household_count] = household_take_measure;
					h_households->warning_area[*h_xmachine_memory_household_count] = household_warning_area;
					h_households->get_warning[*h_xmachine_memory_household_count] = household_get_warning;
					h_households->alert_state[*h_xmachine_memory_household_count] = household_alert_state;
					h_households->sandbag_state[*h_xmachine_memory_household_count] = household_sandbag_state;
					h_households->sandbag_time_count[*h_xmachine_memory_household_count] = household_sandbag_time_count;
					h_households->flooded_time[*h_xmachine_memory_household_count] = household_flooded_time;
					h_households->initial_wl[*h_xmachine_memory_household_count] = household_initial_wl;
					h_households->actual_wl[*h_xmachine_memory_household_count] = household_actual_wl;
					h_households->average_wl[*h_xmachine_memory_household_count] = household_average_wl;
					h_households->max_wl[*h_xmachine_memory_household_count] = household_max_wl;
					h_households->financial_damage[*h_xmachine_memory_household_count] = household_financial_damage;
					h_households->inform_others[*h_xmachine_memory_household_count] = household_inform_others;
					h_households->get_informed[*h_xmachine_memory_household_count] = household_get_informed;
					h_households->lod[*h_xmachine_memory_household_count] = household_lod;
					h_households->animate[*h_xmachine_memory_household_count] = household_animate;
					h_households->animate_dir[*h_xmachine_memory_household_count] = household_animate_dir;
					(*h_xmachine_memory_household_count) ++;	
				}
				else if(strcmp(agentname, "flood") == 0)
				{
					if (*h_xmachine_memory_flood_count > xmachine_memory_flood_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent flood exceeded whilst reading data\n", xmachine_memory_flood_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_floods->x[*h_xmachine_memory_flood_count] = flood_x;//Check maximum x value
                    if(agent_maximum.x < flood_x)
                        agent_maximum.x = (float)flood_x;
                    //Check minimum x value
                    if(agent_minimum.x > flood_x)
                        agent_minimum.x = (float)flood_x;
                    
					h_floods->y[*h_xmachine_memory_flood_count] = flood_y;//Check maximum y value
                    if(agent_maximum.y < flood_y)
                        agent_maximum.y = (float)flood_y;
                    //Check minimum y value
                    if(agent_minimum.y > flood_y)
                        agent_minimum.y = (float)flood_y;
                    
					h_floods->floodID[*h_xmachine_memory_flood_count] = flood_floodID;
					h_floods->flood_h[*h_xmachine_memory_flood_count] = flood_flood_h;
					(*h_xmachine_memory_flood_count) ++;	
				}
				else if(strcmp(agentname, "warning") == 0)
				{
					if (*h_xmachine_memory_warning_count > xmachine_memory_warning_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent warning exceeded whilst reading data\n", xmachine_memory_warning_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_warnings->x[*h_xmachine_memory_warning_count] = warning_x;//Check maximum x value
                    if(agent_maximum.x < warning_x)
                        agent_maximum.x = (float)warning_x;
                    //Check minimum x value
                    if(agent_minimum.x > warning_x)
                        agent_minimum.x = (float)warning_x;
                    
					h_warnings->y[*h_xmachine_memory_warning_count] = warning_y;//Check maximum y value
                    if(agent_maximum.y < warning_y)
                        agent_maximum.y = (float)warning_y;
                    //Check minimum y value
                    if(agent_minimum.y > warning_y)
                        agent_minimum.y = (float)warning_y;
                    
					h_warnings->flooded_households[*h_xmachine_memory_warning_count] = warning_flooded_households;
					h_warnings->total_financial_damage[*h_xmachine_memory_warning_count] = warning_total_financial_damage;
					h_warnings->total_take_measure[*h_xmachine_memory_warning_count] = warning_total_take_measure;
					h_warnings->total_get_warning[*h_xmachine_memory_warning_count] = warning_total_get_warning;
					h_warnings->total_alert_state[*h_xmachine_memory_warning_count] = warning_total_alert_state;
					h_warnings->total_sandbag1[*h_xmachine_memory_warning_count] = warning_total_sandbag1;
					h_warnings->total_sandbag2[*h_xmachine_memory_warning_count] = warning_total_sandbag2;
					h_warnings->total_sandbag3[*h_xmachine_memory_warning_count] = warning_total_sandbag3;
					h_warnings->total_inform_others[*h_xmachine_memory_warning_count] = warning_total_inform_others;
					h_warnings->total_get_informed[*h_xmachine_memory_warning_count] = warning_total_get_informed;
					(*h_xmachine_memory_warning_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                household_x = 0;
                household_y = 0;
                household_resident_num = 0;
                household_OYI = 0;
                household_tenure = 0;
                household_take_measure = 0;
                household_warning_area = 0;
                household_get_warning = 0;
                household_alert_state = 0;
                household_sandbag_state = 0;
                household_sandbag_time_count = 0;
                household_flooded_time = 0;
                household_initial_wl = 0;
                household_actual_wl = 0;
                household_average_wl = 0;
                household_max_wl = 0;
                household_financial_damage = 0;
                household_inform_others = 0;
                household_get_informed = 0;
                household_lod = 0;
                household_animate = 0;
                household_animate_dir = 0;
                flood_x = 0;
                flood_y = 0;
                flood_floodID = 0;
                flood_flood_h = 0;
                warning_x = 0;
                warning_y = 0;
                warning_flooded_households = 0;
                warning_total_financial_damage = 0;
                warning_total_take_measure = 0;
                warning_total_get_warning = 0;
                warning_total_alert_state = 0;
                warning_total_sandbag1 = 0;
                warning_total_sandbag2 = 0;
                warning_total_sandbag3 = 0;
                warning_total_inform_others = 0;
                warning_total_get_informed = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "x") == 0) in_household_x = 1;
			if(strcmp(buffer, "/x") == 0) in_household_x = 0;
			if(strcmp(buffer, "y") == 0) in_household_y = 1;
			if(strcmp(buffer, "/y") == 0) in_household_y = 0;
			if(strcmp(buffer, "resident_num") == 0) in_household_resident_num = 1;
			if(strcmp(buffer, "/resident_num") == 0) in_household_resident_num = 0;
			if(strcmp(buffer, "OYI") == 0) in_household_OYI = 1;
			if(strcmp(buffer, "/OYI") == 0) in_household_OYI = 0;
			if(strcmp(buffer, "tenure") == 0) in_household_tenure = 1;
			if(strcmp(buffer, "/tenure") == 0) in_household_tenure = 0;
			if(strcmp(buffer, "take_measure") == 0) in_household_take_measure = 1;
			if(strcmp(buffer, "/take_measure") == 0) in_household_take_measure = 0;
			if(strcmp(buffer, "warning_area") == 0) in_household_warning_area = 1;
			if(strcmp(buffer, "/warning_area") == 0) in_household_warning_area = 0;
			if(strcmp(buffer, "get_warning") == 0) in_household_get_warning = 1;
			if(strcmp(buffer, "/get_warning") == 0) in_household_get_warning = 0;
			if(strcmp(buffer, "alert_state") == 0) in_household_alert_state = 1;
			if(strcmp(buffer, "/alert_state") == 0) in_household_alert_state = 0;
			if(strcmp(buffer, "sandbag_state") == 0) in_household_sandbag_state = 1;
			if(strcmp(buffer, "/sandbag_state") == 0) in_household_sandbag_state = 0;
			if(strcmp(buffer, "sandbag_time_count") == 0) in_household_sandbag_time_count = 1;
			if(strcmp(buffer, "/sandbag_time_count") == 0) in_household_sandbag_time_count = 0;
			if(strcmp(buffer, "flooded_time") == 0) in_household_flooded_time = 1;
			if(strcmp(buffer, "/flooded_time") == 0) in_household_flooded_time = 0;
			if(strcmp(buffer, "initial_wl") == 0) in_household_initial_wl = 1;
			if(strcmp(buffer, "/initial_wl") == 0) in_household_initial_wl = 0;
			if(strcmp(buffer, "actual_wl") == 0) in_household_actual_wl = 1;
			if(strcmp(buffer, "/actual_wl") == 0) in_household_actual_wl = 0;
			if(strcmp(buffer, "average_wl") == 0) in_household_average_wl = 1;
			if(strcmp(buffer, "/average_wl") == 0) in_household_average_wl = 0;
			if(strcmp(buffer, "max_wl") == 0) in_household_max_wl = 1;
			if(strcmp(buffer, "/max_wl") == 0) in_household_max_wl = 0;
			if(strcmp(buffer, "financial_damage") == 0) in_household_financial_damage = 1;
			if(strcmp(buffer, "/financial_damage") == 0) in_household_financial_damage = 0;
			if(strcmp(buffer, "inform_others") == 0) in_household_inform_others = 1;
			if(strcmp(buffer, "/inform_others") == 0) in_household_inform_others = 0;
			if(strcmp(buffer, "get_informed") == 0) in_household_get_informed = 1;
			if(strcmp(buffer, "/get_informed") == 0) in_household_get_informed = 0;
			if(strcmp(buffer, "lod") == 0) in_household_lod = 1;
			if(strcmp(buffer, "/lod") == 0) in_household_lod = 0;
			if(strcmp(buffer, "animate") == 0) in_household_animate = 1;
			if(strcmp(buffer, "/animate") == 0) in_household_animate = 0;
			if(strcmp(buffer, "animate_dir") == 0) in_household_animate_dir = 1;
			if(strcmp(buffer, "/animate_dir") == 0) in_household_animate_dir = 0;
			if(strcmp(buffer, "x") == 0) in_flood_x = 1;
			if(strcmp(buffer, "/x") == 0) in_flood_x = 0;
			if(strcmp(buffer, "y") == 0) in_flood_y = 1;
			if(strcmp(buffer, "/y") == 0) in_flood_y = 0;
			if(strcmp(buffer, "floodID") == 0) in_flood_floodID = 1;
			if(strcmp(buffer, "/floodID") == 0) in_flood_floodID = 0;
			if(strcmp(buffer, "flood_h") == 0) in_flood_flood_h = 1;
			if(strcmp(buffer, "/flood_h") == 0) in_flood_flood_h = 0;
			if(strcmp(buffer, "x") == 0) in_warning_x = 1;
			if(strcmp(buffer, "/x") == 0) in_warning_x = 0;
			if(strcmp(buffer, "y") == 0) in_warning_y = 1;
			if(strcmp(buffer, "/y") == 0) in_warning_y = 0;
			if(strcmp(buffer, "flooded_households") == 0) in_warning_flooded_households = 1;
			if(strcmp(buffer, "/flooded_households") == 0) in_warning_flooded_households = 0;
			if(strcmp(buffer, "total_financial_damage") == 0) in_warning_total_financial_damage = 1;
			if(strcmp(buffer, "/total_financial_damage") == 0) in_warning_total_financial_damage = 0;
			if(strcmp(buffer, "total_take_measure") == 0) in_warning_total_take_measure = 1;
			if(strcmp(buffer, "/total_take_measure") == 0) in_warning_total_take_measure = 0;
			if(strcmp(buffer, "total_get_warning") == 0) in_warning_total_get_warning = 1;
			if(strcmp(buffer, "/total_get_warning") == 0) in_warning_total_get_warning = 0;
			if(strcmp(buffer, "total_alert_state") == 0) in_warning_total_alert_state = 1;
			if(strcmp(buffer, "/total_alert_state") == 0) in_warning_total_alert_state = 0;
			if(strcmp(buffer, "total_sandbag1") == 0) in_warning_total_sandbag1 = 1;
			if(strcmp(buffer, "/total_sandbag1") == 0) in_warning_total_sandbag1 = 0;
			if(strcmp(buffer, "total_sandbag2") == 0) in_warning_total_sandbag2 = 1;
			if(strcmp(buffer, "/total_sandbag2") == 0) in_warning_total_sandbag2 = 0;
			if(strcmp(buffer, "total_sandbag3") == 0) in_warning_total_sandbag3 = 1;
			if(strcmp(buffer, "/total_sandbag3") == 0) in_warning_total_sandbag3 = 0;
			if(strcmp(buffer, "total_inform_others") == 0) in_warning_total_inform_others = 1;
			if(strcmp(buffer, "/total_inform_others") == 0) in_warning_total_inform_others = 0;
			if(strcmp(buffer, "total_get_informed") == 0) in_warning_total_get_informed = 1;
			if(strcmp(buffer, "/total_get_informed") == 0) in_warning_total_get_informed = 0;
			
            /* environment variables */
            if(strcmp(buffer, "FLOOD_DATA_ARRAY") == 0) in_env_FLOOD_DATA_ARRAY = 1;
            if(strcmp(buffer, "/FLOOD_DATA_ARRAY") == 0) in_env_FLOOD_DATA_ARRAY = 0;
			if(strcmp(buffer, "TIME") == 0) in_env_TIME = 1;
            if(strcmp(buffer, "/TIME") == 0) in_env_TIME = 0;
			if(strcmp(buffer, "RANDOM_SEED_SEC") == 0) in_env_RANDOM_SEED_SEC = 1;
            if(strcmp(buffer, "/RANDOM_SEED_SEC") == 0) in_env_RANDOM_SEED_SEC = 0;
			if(strcmp(buffer, "RANDOM_SEED_MIN") == 0) in_env_RANDOM_SEED_MIN = 1;
            if(strcmp(buffer, "/RANDOM_SEED_MIN") == 0) in_env_RANDOM_SEED_MIN = 0;
			if(strcmp(buffer, "TIME_SCALER") == 0) in_env_TIME_SCALER = 1;
            if(strcmp(buffer, "/TIME_SCALER") == 0) in_env_TIME_SCALER = 0;
			

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_household_x){
                    household_x = (float) fgpu_atof(buffer); 
                }
				if(in_household_y){
                    household_y = (float) fgpu_atof(buffer); 
                }
				if(in_household_resident_num){
                    household_resident_num = (int) fpgu_strtol(buffer); 
                }
				if(in_household_OYI){
                    household_OYI = (int) fpgu_strtol(buffer); 
                }
				if(in_household_tenure){
                    household_tenure = (int) fpgu_strtol(buffer); 
                }
				if(in_household_take_measure){
                    household_take_measure = (int) fpgu_strtol(buffer); 
                }
				if(in_household_warning_area){
                    household_warning_area = (int) fpgu_strtol(buffer); 
                }
				if(in_household_get_warning){
                    household_get_warning = (int) fpgu_strtol(buffer); 
                }
				if(in_household_alert_state){
                    household_alert_state = (int) fpgu_strtol(buffer); 
                }
				if(in_household_sandbag_state){
                    household_sandbag_state = (int) fpgu_strtol(buffer); 
                }
				if(in_household_sandbag_time_count){
                    household_sandbag_time_count = (int) fpgu_strtol(buffer); 
                }
				if(in_household_flooded_time){
                    household_flooded_time = (int) fpgu_strtol(buffer); 
                }
				if(in_household_initial_wl){
                    household_initial_wl = (float) fgpu_atof(buffer); 
                }
				if(in_household_actual_wl){
                    household_actual_wl = (float) fgpu_atof(buffer); 
                }
				if(in_household_average_wl){
                    household_average_wl = (float) fgpu_atof(buffer); 
                }
				if(in_household_max_wl){
                    household_max_wl = (float) fgpu_atof(buffer); 
                }
				if(in_household_financial_damage){
                    household_financial_damage = (float) fgpu_atof(buffer); 
                }
				if(in_household_inform_others){
                    household_inform_others = (int) fpgu_strtol(buffer); 
                }
				if(in_household_get_informed){
                    household_get_informed = (int) fpgu_strtol(buffer); 
                }
				if(in_household_lod){
                    household_lod = (int) fpgu_strtol(buffer); 
                }
				if(in_household_animate){
                    household_animate = (float) fgpu_atof(buffer); 
                }
				if(in_household_animate_dir){
                    household_animate_dir = (int) fpgu_strtol(buffer); 
                }
				if(in_flood_x){
                    flood_x = (int) fpgu_strtol(buffer); 
                }
				if(in_flood_y){
                    flood_y = (int) fpgu_strtol(buffer); 
                }
				if(in_flood_floodID){
                    flood_floodID = (int) fpgu_strtol(buffer); 
                }
				if(in_flood_flood_h){
                    flood_flood_h = (float) fgpu_atof(buffer); 
                }
				if(in_warning_x){
                    warning_x = (float) fgpu_atof(buffer); 
                }
				if(in_warning_y){
                    warning_y = (float) fgpu_atof(buffer); 
                }
				if(in_warning_flooded_households){
                    warning_flooded_households = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_financial_damage){
                    warning_total_financial_damage = (float) fgpu_atof(buffer); 
                }
				if(in_warning_total_take_measure){
                    warning_total_take_measure = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_get_warning){
                    warning_total_get_warning = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_alert_state){
                    warning_total_alert_state = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_sandbag1){
                    warning_total_sandbag1 = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_sandbag2){
                    warning_total_sandbag2 = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_sandbag3){
                    warning_total_sandbag3 = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_inform_others){
                    warning_total_inform_others = (int) fpgu_strtol(buffer); 
                }
				if(in_warning_total_get_informed){
                    warning_total_get_informed = (int) fpgu_strtol(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_FLOOD_DATA_ARRAY){
              readArrayInput<short>(&fpgu_strtol, buffer, env_FLOOD_DATA_ARRAY, 900000);
                    set_FLOOD_DATA_ARRAY(env_FLOOD_DATA_ARRAY);
                  
              }
            if(in_env_TIME){
              
                    env_TIME = (int) fpgu_strtol(buffer);
                    
                    set_TIME(&env_TIME);
                  
              }
            if(in_env_RANDOM_SEED_SEC){
              
                    env_RANDOM_SEED_SEC = (int) fpgu_strtol(buffer);
                    
                    set_RANDOM_SEED_SEC(&env_RANDOM_SEED_SEC);
                  
              }
            if(in_env_RANDOM_SEED_MIN){
              
                    env_RANDOM_SEED_MIN = (int) fpgu_strtol(buffer);
                    
                    set_RANDOM_SEED_MIN(&env_RANDOM_SEED_MIN);
                  
              }
            if(in_env_TIME_SCALER){
              
                    env_TIME_SCALER = (float) fgpu_atof(buffer);
                    
                    set_TIME_SCALER(&env_TIME_SCALER);
                  
              }
            
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 && c == '-' && buffer[1] == '-' && buffer[0] == '!'){
	in_comment = 1;
	// Reset the buffer and i.
	buffer[0] = 0;
	i = 0;
	}

	// Store the character and increment the counter
	buffer[i] = c;
	i++;

	}
	/* If in data read char into buffer */
	else
	{
	buffer[i] = c;
	i++;
	}
	}
	// If no bytes were read, raise a warning.
	if(bytesRead == 0){
	fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
	fflush(stdout);
	}

	// If the in_comment flag is still marked, issue a warning.
	if(in_comment){
	fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
	fflush(stdout);
	}

	/* Close the file */
	fclose(file);

	/* Variables for environment variables */
	
				// short env_FLOOD_DATA_ARRAY[900000];

				delete [] env_FLOOD_DATA_ARRAY;
			
	
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}


/* Methods to load static networks from disk */
