
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


#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_household_count;

__constant__ int d_xmachine_memory_flood_count;

__constant__ int d_xmachine_memory_warning_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_household_default_count;

__constant__ int d_xmachine_memory_flood_static_count;

__constant__ int d_xmachine_memory_warning_static_warning_count;


/* Message constants */

/* flood_cell Message variables */
//Discrete Partitioning Variables
__constant__ int d_message_flood_cell_range;     /**< range of the discrete message*/
__constant__ int d_message_flood_cell_width;     /**< with of the message grid*/

/* financial_damage_infor Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_financial_damage_infor_count;         /**< message list counter*/
__constant__ int d_message_financial_damage_infor_output_type;   /**< message output type (single or optional)*/

/* state_data Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_state_data_count;         /**< message list counter*/
__constant__ int d_message_state_data_output_type;   /**< message output type (single or optional)*/

	

/* Graph Constants */


/* Graph device array pointer(s) */


/* Graph host array pointer(s) */

    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* flood_cell Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_flood_cell_x;
__constant__ int d_tex_xmachine_message_flood_cell_x_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_flood_cell_y;
__constant__ int d_tex_xmachine_message_flood_cell_y_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_flood_cell_floodID;
__constant__ int d_tex_xmachine_message_flood_cell_floodID_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_flood_cell_flood_h;
__constant__ int d_tex_xmachine_message_flood_cell_flood_h_offset;



    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created household agent functions */

/** reset_household_scan_input
 * household agent reset scan input function
 * @param agents The xmachine_memory_household_list agent list
 */
__global__ void reset_household_scan_input(xmachine_memory_household_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_household_Agents
 * household scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_household_list agent list destination
 * @param agents_src xmachine_memory_household_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_household_Agents(xmachine_memory_household_list* agents_dst, xmachine_memory_household_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->resident_num[output_index] = agents_src->resident_num[index];        
		agents_dst->OYI[output_index] = agents_src->OYI[index];        
		agents_dst->tenure[output_index] = agents_src->tenure[index];        
		agents_dst->take_measure[output_index] = agents_src->take_measure[index];        
		agents_dst->warning_area[output_index] = agents_src->warning_area[index];        
		agents_dst->get_warning[output_index] = agents_src->get_warning[index];        
		agents_dst->alert_state[output_index] = agents_src->alert_state[index];        
		agents_dst->sandbag_state[output_index] = agents_src->sandbag_state[index];        
		agents_dst->sandbag_time_count[output_index] = agents_src->sandbag_time_count[index];        
		agents_dst->flooded_time[output_index] = agents_src->flooded_time[index];        
		agents_dst->initial_wl[output_index] = agents_src->initial_wl[index];        
		agents_dst->actual_wl[output_index] = agents_src->actual_wl[index];        
		agents_dst->average_wl[output_index] = agents_src->average_wl[index];        
		agents_dst->max_wl[output_index] = agents_src->max_wl[index];        
		agents_dst->financial_damage[output_index] = agents_src->financial_damage[index];        
		agents_dst->inform_others[output_index] = agents_src->inform_others[index];        
		agents_dst->get_informed[output_index] = agents_src->get_informed[index];        
		agents_dst->lod[output_index] = agents_src->lod[index];        
		agents_dst->animate[output_index] = agents_src->animate[index];        
		agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];
	}
}

/** append_household_Agents
 * household scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_household_list agent list destination
 * @param agents_src xmachine_memory_household_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_household_Agents(xmachine_memory_household_list* agents_dst, xmachine_memory_household_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->resident_num[output_index] = agents_src->resident_num[index];
	    agents_dst->OYI[output_index] = agents_src->OYI[index];
	    agents_dst->tenure[output_index] = agents_src->tenure[index];
	    agents_dst->take_measure[output_index] = agents_src->take_measure[index];
	    agents_dst->warning_area[output_index] = agents_src->warning_area[index];
	    agents_dst->get_warning[output_index] = agents_src->get_warning[index];
	    agents_dst->alert_state[output_index] = agents_src->alert_state[index];
	    agents_dst->sandbag_state[output_index] = agents_src->sandbag_state[index];
	    agents_dst->sandbag_time_count[output_index] = agents_src->sandbag_time_count[index];
	    agents_dst->flooded_time[output_index] = agents_src->flooded_time[index];
	    agents_dst->initial_wl[output_index] = agents_src->initial_wl[index];
	    agents_dst->actual_wl[output_index] = agents_src->actual_wl[index];
	    agents_dst->average_wl[output_index] = agents_src->average_wl[index];
	    agents_dst->max_wl[output_index] = agents_src->max_wl[index];
	    agents_dst->financial_damage[output_index] = agents_src->financial_damage[index];
	    agents_dst->inform_others[output_index] = agents_src->inform_others[index];
	    agents_dst->get_informed[output_index] = agents_src->get_informed[index];
	    agents_dst->lod[output_index] = agents_src->lod[index];
	    agents_dst->animate[output_index] = agents_src->animate[index];
	    agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];
    }
}

/** add_household_agent
 * Continuous household agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_household_list to add agents to 
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param resident_num agent variable of type int
 * @param OYI agent variable of type int
 * @param tenure agent variable of type int
 * @param take_measure agent variable of type int
 * @param warning_area agent variable of type int
 * @param get_warning agent variable of type int
 * @param alert_state agent variable of type int
 * @param sandbag_state agent variable of type int
 * @param sandbag_time_count agent variable of type int
 * @param flooded_time agent variable of type int
 * @param initial_wl agent variable of type float
 * @param actual_wl agent variable of type float
 * @param average_wl agent variable of type float
 * @param max_wl agent variable of type float
 * @param financial_damage agent variable of type float
 * @param inform_others agent variable of type int
 * @param get_informed agent variable of type int
 * @param lod agent variable of type int
 * @param animate agent variable of type float
 * @param animate_dir agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_household_agent(xmachine_memory_household_list* agents, float x, float y, int resident_num, int OYI, int tenure, int take_measure, int warning_area, int get_warning, int alert_state, int sandbag_state, int sandbag_time_count, int flooded_time, float initial_wl, float actual_wl, float average_wl, float max_wl, float financial_damage, int inform_others, int get_informed, int lod, float animate, int animate_dir){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->x[index] = x;
	agents->y[index] = y;
	agents->resident_num[index] = resident_num;
	agents->OYI[index] = OYI;
	agents->tenure[index] = tenure;
	agents->take_measure[index] = take_measure;
	agents->warning_area[index] = warning_area;
	agents->get_warning[index] = get_warning;
	agents->alert_state[index] = alert_state;
	agents->sandbag_state[index] = sandbag_state;
	agents->sandbag_time_count[index] = sandbag_time_count;
	agents->flooded_time[index] = flooded_time;
	agents->initial_wl[index] = initial_wl;
	agents->actual_wl[index] = actual_wl;
	agents->average_wl[index] = average_wl;
	agents->max_wl[index] = max_wl;
	agents->financial_damage[index] = financial_damage;
	agents->inform_others[index] = inform_others;
	agents->get_informed[index] = get_informed;
	agents->lod[index] = lod;
	agents->animate[index] = animate;
	agents->animate_dir[index] = animate_dir;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_household_agent(xmachine_memory_household_list* agents, float x, float y, int resident_num, int OYI, int tenure, int take_measure, int warning_area, int get_warning, int alert_state, int sandbag_state, int sandbag_time_count, int flooded_time, float initial_wl, float actual_wl, float average_wl, float max_wl, float financial_damage, int inform_others, int get_informed, int lod, float animate, int animate_dir){
    add_household_agent<DISCRETE_2D>(agents, x, y, resident_num, OYI, tenure, take_measure, warning_area, get_warning, alert_state, sandbag_state, sandbag_time_count, flooded_time, initial_wl, actual_wl, average_wl, max_wl, financial_damage, inform_others, get_informed, lod, animate, animate_dir);
}

/** reorder_household_agents
 * Continuous household agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_household_agents(unsigned int* values, xmachine_memory_household_list* unordered_agents, xmachine_memory_household_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->resident_num[index] = unordered_agents->resident_num[old_pos];
	ordered_agents->OYI[index] = unordered_agents->OYI[old_pos];
	ordered_agents->tenure[index] = unordered_agents->tenure[old_pos];
	ordered_agents->take_measure[index] = unordered_agents->take_measure[old_pos];
	ordered_agents->warning_area[index] = unordered_agents->warning_area[old_pos];
	ordered_agents->get_warning[index] = unordered_agents->get_warning[old_pos];
	ordered_agents->alert_state[index] = unordered_agents->alert_state[old_pos];
	ordered_agents->sandbag_state[index] = unordered_agents->sandbag_state[old_pos];
	ordered_agents->sandbag_time_count[index] = unordered_agents->sandbag_time_count[old_pos];
	ordered_agents->flooded_time[index] = unordered_agents->flooded_time[old_pos];
	ordered_agents->initial_wl[index] = unordered_agents->initial_wl[old_pos];
	ordered_agents->actual_wl[index] = unordered_agents->actual_wl[old_pos];
	ordered_agents->average_wl[index] = unordered_agents->average_wl[old_pos];
	ordered_agents->max_wl[index] = unordered_agents->max_wl[old_pos];
	ordered_agents->financial_damage[index] = unordered_agents->financial_damage[old_pos];
	ordered_agents->inform_others[index] = unordered_agents->inform_others[old_pos];
	ordered_agents->get_informed[index] = unordered_agents->get_informed[old_pos];
	ordered_agents->lod[index] = unordered_agents->lod[old_pos];
	ordered_agents->animate[index] = unordered_agents->animate[old_pos];
	ordered_agents->animate_dir[index] = unordered_agents->animate_dir[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created flood agent functions */

/** reset_flood_scan_input
 * flood agent reset scan input function
 * @param agents The xmachine_memory_flood_list agent list
 */
__global__ void reset_flood_scan_input(xmachine_memory_flood_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created warning agent functions */

/** reset_warning_scan_input
 * warning agent reset scan input function
 * @param agents The xmachine_memory_warning_list agent list
 */
__global__ void reset_warning_scan_input(xmachine_memory_warning_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_warning_Agents
 * warning scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_warning_list agent list destination
 * @param agents_src xmachine_memory_warning_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_warning_Agents(xmachine_memory_warning_list* agents_dst, xmachine_memory_warning_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->flooded_households[output_index] = agents_src->flooded_households[index];        
		agents_dst->total_financial_damage[output_index] = agents_src->total_financial_damage[index];        
		agents_dst->total_take_measure[output_index] = agents_src->total_take_measure[index];        
		agents_dst->total_get_warning[output_index] = agents_src->total_get_warning[index];        
		agents_dst->total_alert_state[output_index] = agents_src->total_alert_state[index];        
		agents_dst->total_sandbag1[output_index] = agents_src->total_sandbag1[index];        
		agents_dst->total_sandbag2[output_index] = agents_src->total_sandbag2[index];        
		agents_dst->total_sandbag3[output_index] = agents_src->total_sandbag3[index];        
		agents_dst->total_inform_others[output_index] = agents_src->total_inform_others[index];        
		agents_dst->total_get_informed[output_index] = agents_src->total_get_informed[index];
	}
}

/** append_warning_Agents
 * warning scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_warning_list agent list destination
 * @param agents_src xmachine_memory_warning_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_warning_Agents(xmachine_memory_warning_list* agents_dst, xmachine_memory_warning_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->flooded_households[output_index] = agents_src->flooded_households[index];
	    agents_dst->total_financial_damage[output_index] = agents_src->total_financial_damage[index];
	    agents_dst->total_take_measure[output_index] = agents_src->total_take_measure[index];
	    agents_dst->total_get_warning[output_index] = agents_src->total_get_warning[index];
	    agents_dst->total_alert_state[output_index] = agents_src->total_alert_state[index];
	    agents_dst->total_sandbag1[output_index] = agents_src->total_sandbag1[index];
	    agents_dst->total_sandbag2[output_index] = agents_src->total_sandbag2[index];
	    agents_dst->total_sandbag3[output_index] = agents_src->total_sandbag3[index];
	    agents_dst->total_inform_others[output_index] = agents_src->total_inform_others[index];
	    agents_dst->total_get_informed[output_index] = agents_src->total_get_informed[index];
    }
}

/** add_warning_agent
 * Continuous warning agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_warning_list to add agents to 
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param flooded_households agent variable of type int
 * @param total_financial_damage agent variable of type float
 * @param total_take_measure agent variable of type int
 * @param total_get_warning agent variable of type int
 * @param total_alert_state agent variable of type int
 * @param total_sandbag1 agent variable of type int
 * @param total_sandbag2 agent variable of type int
 * @param total_sandbag3 agent variable of type int
 * @param total_inform_others agent variable of type int
 * @param total_get_informed agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_warning_agent(xmachine_memory_warning_list* agents, float x, float y, int flooded_households, float total_financial_damage, int total_take_measure, int total_get_warning, int total_alert_state, int total_sandbag1, int total_sandbag2, int total_sandbag3, int total_inform_others, int total_get_informed){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->x[index] = x;
	agents->y[index] = y;
	agents->flooded_households[index] = flooded_households;
	agents->total_financial_damage[index] = total_financial_damage;
	agents->total_take_measure[index] = total_take_measure;
	agents->total_get_warning[index] = total_get_warning;
	agents->total_alert_state[index] = total_alert_state;
	agents->total_sandbag1[index] = total_sandbag1;
	agents->total_sandbag2[index] = total_sandbag2;
	agents->total_sandbag3[index] = total_sandbag3;
	agents->total_inform_others[index] = total_inform_others;
	agents->total_get_informed[index] = total_get_informed;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_warning_agent(xmachine_memory_warning_list* agents, float x, float y, int flooded_households, float total_financial_damage, int total_take_measure, int total_get_warning, int total_alert_state, int total_sandbag1, int total_sandbag2, int total_sandbag3, int total_inform_others, int total_get_informed){
    add_warning_agent<DISCRETE_2D>(agents, x, y, flooded_households, total_financial_damage, total_take_measure, total_get_warning, total_alert_state, total_sandbag1, total_sandbag2, total_sandbag3, total_inform_others, total_get_informed);
}

/** reorder_warning_agents
 * Continuous warning agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_warning_agents(unsigned int* values, xmachine_memory_warning_list* unordered_agents, xmachine_memory_warning_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->flooded_households[index] = unordered_agents->flooded_households[old_pos];
	ordered_agents->total_financial_damage[index] = unordered_agents->total_financial_damage[old_pos];
	ordered_agents->total_take_measure[index] = unordered_agents->total_take_measure[old_pos];
	ordered_agents->total_get_warning[index] = unordered_agents->total_get_warning[old_pos];
	ordered_agents->total_alert_state[index] = unordered_agents->total_alert_state[old_pos];
	ordered_agents->total_sandbag1[index] = unordered_agents->total_sandbag1[old_pos];
	ordered_agents->total_sandbag2[index] = unordered_agents->total_sandbag2[old_pos];
	ordered_agents->total_sandbag3[index] = unordered_agents->total_sandbag3[old_pos];
	ordered_agents->total_inform_others[index] = unordered_agents->total_inform_others[old_pos];
	ordered_agents->total_get_informed[index] = unordered_agents->total_get_informed[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created flood_cell message functions */


/* Message functions */

template <int AGENT_TYPE>
__device__ void add_flood_cell_message(xmachine_message_flood_cell_list* messages, int x, int y, int floodID, float flood_h){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		
		messages->x[index] = x;			
		messages->y[index] = y;			
		messages->floodID[index] = floodID;			
		messages->flood_h[index] = flood_h;			
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_flood_cell* get_first_flood_cell_message_continuous(xmachine_message_flood_cell_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_flood_cell* message_share = (xmachine_message_flood_cell*)&sm_data[0];
	
	int range = d_message_flood_cell_range;
	int width = d_message_flood_cell_width;

	glm::ivec2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);

	int index = ((global_position.y)* width) + global_position.x;

	xmachine_message_flood_cell temp_message;
	temp_message._position = glm::ivec2(agent_x, agent_y);
	temp_message._relative = glm::ivec2(-range, -range);

	temp_message.x = tex1Dfetch(tex_xmachine_message_flood_cell_x, index + d_tex_xmachine_message_flood_cell_x_offset);temp_message.y = tex1Dfetch(tex_xmachine_message_flood_cell_y, index + d_tex_xmachine_message_flood_cell_y_offset);temp_message.floodID = tex1Dfetch(tex_xmachine_message_flood_cell_floodID, index + d_tex_xmachine_message_flood_cell_floodID_offset);temp_message.flood_h = tex1Dfetch(tex_xmachine_message_flood_cell_flood_h, index + d_tex_xmachine_message_flood_cell_flood_h_offset);
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &message_share[threadIdx.x];
}

//Get next flood_cell message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_flood_cell* get_next_flood_cell_message_continuous(xmachine_message_flood_cell* message, xmachine_message_flood_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_flood_cell* message_share = (xmachine_message_flood_cell*)&sm_data[0];
	
	int range = d_message_flood_cell_range;
	int width = d_message_flood_cell_width;

	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	glm::ivec2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_flood_cell temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	temp_message.x = tex1Dfetch(tex_xmachine_message_flood_cell_x, index + d_tex_xmachine_message_flood_cell_x_offset);	temp_message.y = tex1Dfetch(tex_xmachine_message_flood_cell_y, index + d_tex_xmachine_message_flood_cell_y_offset);	temp_message.floodID = tex1Dfetch(tex_xmachine_message_flood_cell_floodID, index + d_tex_xmachine_message_flood_cell_floodID_offset);	temp_message.flood_h = tex1Dfetch(tex_xmachine_message_flood_cell_flood_h, index + d_tex_xmachine_message_flood_cell_flood_h_offset);	

	message_share[threadIdx.x] = temp_message;

	return &message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void flood_cell_message_to_sm(xmachine_message_flood_cell_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_flood_cell temp_message;
		
		temp_message.x = messages->x[global_index];		
		temp_message.y = messages->y[global_index];		
		temp_message.floodID = messages->floodID[global_index];		
		temp_message.flood_h = messages->flood_h[global_index];		

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_flood_cell));
	  xmachine_message_flood_cell* sm_message = ((xmachine_message_flood_cell*)&message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first flood_cell message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergences
__device__ xmachine_message_flood_cell* get_first_flood_cell_message_discrete(xmachine_message_flood_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();

	int range = d_message_flood_cell_range;
	int width = d_message_flood_cell_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memory of first load
	glm::ivec2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memory (coalesced read)
	flood_cell_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x < range);
	int right_border = (threadIdx.x >= (blockDim.x-range));
	int top_border = (threadIdx.y < range);
	int bottom_border = (threadIdx.y >= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		flood_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_flood_cell));
	xmachine_message_flood_cell* temp = ((xmachine_message_flood_cell*)&message_share[message_index]);
	temp->_relative = glm::ivec2(-range, -range); //this is the relative position
	return temp;
}

//Get next flood_cell message 
//Used by discrete agents this accesses messages through shared memory which were all loaded on first message retrieval call.
__device__ xmachine_message_flood_cell* get_next_flood_cell_message_discrete(xmachine_message_flood_cell* message, xmachine_message_flood_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();
	
	int range = d_message_flood_cell_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	glm::ivec2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_flood_cell));
	xmachine_message_flood_cell* temp = ((xmachine_message_flood_cell*)&message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first flood_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_flood_cell* get_first_flood_cell_message(xmachine_message_flood_cell_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_flood_cell_message_discrete(messages);
	else	//use texture fetching method
		return get_first_flood_cell_message_continuous(messages, agent_x, agent_y);

}

//Get next flood_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_flood_cell* get_next_flood_cell_message(xmachine_message_flood_cell* message, xmachine_message_flood_cell_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_flood_cell_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_flood_cell_message_continuous(message, messages);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created financial_damage_infor message functions */


/** add_financial_damage_infor_message
 * Add non partitioned or spatially partitioned financial_damage_infor message
 * @param messages xmachine_message_financial_damage_infor_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param max_wl agent variable of type float
 * @param financial_damage agent variable of type float
 * @param take_measure agent variable of type int
 * @param get_warning agent variable of type int
 * @param alert_state agent variable of type int
 * @param sandbag_state agent variable of type int
 * @param inform_others agent variable of type int
 * @param get_informed agent variable of type int
 * @param flooded_time agent variable of type int
 * @param actual_wl agent variable of type float
 */
__device__ void add_financial_damage_infor_message(xmachine_message_financial_damage_infor_list* messages, float x, float y, float z, float max_wl, float financial_damage, int take_measure, int get_warning, int alert_state, int sandbag_state, int inform_others, int get_informed, int flooded_time, float actual_wl){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_financial_damage_infor_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_financial_damage_infor_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_financial_damage_infor_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_financial_damage_infor Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->max_wl[index] = max_wl;
	messages->financial_damage[index] = financial_damage;
	messages->take_measure[index] = take_measure;
	messages->get_warning[index] = get_warning;
	messages->alert_state[index] = alert_state;
	messages->sandbag_state[index] = sandbag_state;
	messages->inform_others[index] = inform_others;
	messages->get_informed[index] = get_informed;
	messages->flooded_time[index] = flooded_time;
	messages->actual_wl[index] = actual_wl;

}

/**
 * Scatter non partitioned or spatially partitioned financial_damage_infor message (for optional messages)
 * @param messages scatter_optional_financial_damage_infor_messages Sparse xmachine_message_financial_damage_infor_list message list
 * @param message_swap temp xmachine_message_financial_damage_infor_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_financial_damage_infor_messages(xmachine_message_financial_damage_infor_list* messages, xmachine_message_financial_damage_infor_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_financial_damage_infor_count;

		//AoS - xmachine_message_financial_damage_infor Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->max_wl[output_index] = messages_swap->max_wl[index];
		messages->financial_damage[output_index] = messages_swap->financial_damage[index];
		messages->take_measure[output_index] = messages_swap->take_measure[index];
		messages->get_warning[output_index] = messages_swap->get_warning[index];
		messages->alert_state[output_index] = messages_swap->alert_state[index];
		messages->sandbag_state[output_index] = messages_swap->sandbag_state[index];
		messages->inform_others[output_index] = messages_swap->inform_others[index];
		messages->get_informed[output_index] = messages_swap->get_informed[index];
		messages->flooded_time[output_index] = messages_swap->flooded_time[index];
		messages->actual_wl[output_index] = messages_swap->actual_wl[index];				
	}
}

/** reset_financial_damage_infor_swaps
 * Reset non partitioned or spatially partitioned financial_damage_infor message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_financial_damage_infor_swaps(xmachine_message_financial_damage_infor_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_financial_damage_infor* get_first_financial_damage_infor_message(xmachine_message_financial_damage_infor_list* messages){
	
	//printf("call continuous financial_damage_infor");
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_financial_damage_infor_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_financial_damage_infor Coalesced memory read
	xmachine_message_financial_damage_infor temp_message;
	temp_message._position = messages->_position[index];
	temp_message.x = messages->x[index];
	temp_message.y = messages->y[index];
	temp_message.z = messages->z[index];
	temp_message.max_wl = messages->max_wl[index];
	temp_message.financial_damage = messages->financial_damage[index];
	temp_message.take_measure = messages->take_measure[index];
	temp_message.get_warning = messages->get_warning[index];
	temp_message.alert_state = messages->alert_state[index];
	temp_message.sandbag_state = messages->sandbag_state[index];
	temp_message.inform_others = messages->inform_others[index];
	temp_message.get_informed = messages->get_informed[index];
	temp_message.flooded_time = messages->flooded_time[index];
	temp_message.actual_wl = messages->actual_wl[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_financial_damage_infor));
	xmachine_message_financial_damage_infor* sm_message = ((xmachine_message_financial_damage_infor*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_financial_damage_infor*)&message_share[d_SM_START]);
}

__device__ xmachine_message_financial_damage_infor* get_next_financial_damage_infor_message(xmachine_message_financial_damage_infor* message, xmachine_message_financial_damage_infor_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_financial_damage_infor_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_financial_damage_infor_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_financial_damage_infor Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_financial_damage_infor temp_message;
		temp_message._position = messages->_position[index];
		temp_message.x = messages->x[index];
		temp_message.y = messages->y[index];
		temp_message.z = messages->z[index];
		temp_message.max_wl = messages->max_wl[index];
		temp_message.financial_damage = messages->financial_damage[index];
		temp_message.take_measure = messages->take_measure[index];
		temp_message.get_warning = messages->get_warning[index];
		temp_message.alert_state = messages->alert_state[index];
		temp_message.sandbag_state = messages->sandbag_state[index];
		temp_message.inform_others = messages->inform_others[index];
		temp_message.get_informed = messages->get_informed[index];
		temp_message.flooded_time = messages->flooded_time[index];
		temp_message.actual_wl = messages->actual_wl[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_financial_damage_infor));
		xmachine_message_financial_damage_infor* sm_message = ((xmachine_message_financial_damage_infor*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_financial_damage_infor));
	return ((xmachine_message_financial_damage_infor*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created state_data message functions */


/** add_state_data_message
 * Add non partitioned or spatially partitioned state_data message
 * @param messages xmachine_message_state_data_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param flooded_households agent variable of type int
 * @param total_financial_damage agent variable of type float
 * @param total_take_measure agent variable of type int
 * @param total_get_warning agent variable of type int
 * @param total_alert_state agent variable of type int
 * @param total_sandbag1 agent variable of type int
 * @param total_sandbag2 agent variable of type int
 * @param total_sandbag3 agent variable of type int
 * @param total_inform_others agent variable of type int
 * @param total_get_informed agent variable of type int
 */
__device__ void add_state_data_message(xmachine_message_state_data_list* messages, float x, float y, int flooded_households, float total_financial_damage, int total_take_measure, int total_get_warning, int total_alert_state, int total_sandbag1, int total_sandbag2, int total_sandbag3, int total_inform_others, int total_get_informed){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_state_data_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_state_data_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_state_data_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_state_data Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->flooded_households[index] = flooded_households;
	messages->total_financial_damage[index] = total_financial_damage;
	messages->total_take_measure[index] = total_take_measure;
	messages->total_get_warning[index] = total_get_warning;
	messages->total_alert_state[index] = total_alert_state;
	messages->total_sandbag1[index] = total_sandbag1;
	messages->total_sandbag2[index] = total_sandbag2;
	messages->total_sandbag3[index] = total_sandbag3;
	messages->total_inform_others[index] = total_inform_others;
	messages->total_get_informed[index] = total_get_informed;

}

/**
 * Scatter non partitioned or spatially partitioned state_data message (for optional messages)
 * @param messages scatter_optional_state_data_messages Sparse xmachine_message_state_data_list message list
 * @param message_swap temp xmachine_message_state_data_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_state_data_messages(xmachine_message_state_data_list* messages, xmachine_message_state_data_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_state_data_count;

		//AoS - xmachine_message_state_data Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->flooded_households[output_index] = messages_swap->flooded_households[index];
		messages->total_financial_damage[output_index] = messages_swap->total_financial_damage[index];
		messages->total_take_measure[output_index] = messages_swap->total_take_measure[index];
		messages->total_get_warning[output_index] = messages_swap->total_get_warning[index];
		messages->total_alert_state[output_index] = messages_swap->total_alert_state[index];
		messages->total_sandbag1[output_index] = messages_swap->total_sandbag1[index];
		messages->total_sandbag2[output_index] = messages_swap->total_sandbag2[index];
		messages->total_sandbag3[output_index] = messages_swap->total_sandbag3[index];
		messages->total_inform_others[output_index] = messages_swap->total_inform_others[index];
		messages->total_get_informed[output_index] = messages_swap->total_get_informed[index];				
	}
}

/** reset_state_data_swaps
 * Reset non partitioned or spatially partitioned state_data message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_state_data_swaps(xmachine_message_state_data_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_state_data* get_first_state_data_message(xmachine_message_state_data_list* messages){
	
	//printf("call continuous state_data");
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_state_data_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_state_data Coalesced memory read
	xmachine_message_state_data temp_message;
	temp_message._position = messages->_position[index];
	temp_message.x = messages->x[index];
	temp_message.y = messages->y[index];
	temp_message.flooded_households = messages->flooded_households[index];
	temp_message.total_financial_damage = messages->total_financial_damage[index];
	temp_message.total_take_measure = messages->total_take_measure[index];
	temp_message.total_get_warning = messages->total_get_warning[index];
	temp_message.total_alert_state = messages->total_alert_state[index];
	temp_message.total_sandbag1 = messages->total_sandbag1[index];
	temp_message.total_sandbag2 = messages->total_sandbag2[index];
	temp_message.total_sandbag3 = messages->total_sandbag3[index];
	temp_message.total_inform_others = messages->total_inform_others[index];
	temp_message.total_get_informed = messages->total_get_informed[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_state_data));
	xmachine_message_state_data* sm_message = ((xmachine_message_state_data*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_state_data*)&message_share[d_SM_START]);
}

__device__ xmachine_message_state_data* get_next_state_data_message(xmachine_message_state_data* message, xmachine_message_state_data_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_state_data_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_state_data_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_state_data Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_state_data temp_message;
		temp_message._position = messages->_position[index];
		temp_message.x = messages->x[index];
		temp_message.y = messages->y[index];
		temp_message.flooded_households = messages->flooded_households[index];
		temp_message.total_financial_damage = messages->total_financial_damage[index];
		temp_message.total_take_measure = messages->total_take_measure[index];
		temp_message.total_get_warning = messages->total_get_warning[index];
		temp_message.total_alert_state = messages->total_alert_state[index];
		temp_message.total_sandbag1 = messages->total_sandbag1[index];
		temp_message.total_sandbag2 = messages->total_sandbag2[index];
		temp_message.total_sandbag3 = messages->total_sandbag3[index];
		temp_message.total_inform_others = messages->total_inform_others[index];
		temp_message.total_get_informed = messages->total_get_informed[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_state_data));
		xmachine_message_state_data* sm_message = ((xmachine_message_state_data*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_state_data));
	return ((xmachine_message_state_data*)&message_share[message_index]);
}

	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */


/**
 *
 */


__global__ void GPUFLAME_output_financial_damage_infor(xmachine_memory_household_list* agents, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_household_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_financial_damage_infor Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_household agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.resident_num = agents->resident_num[index];
	agent.OYI = agents->OYI[index];
	agent.tenure = agents->tenure[index];
	agent.take_measure = agents->take_measure[index];
	agent.warning_area = agents->warning_area[index];
	agent.get_warning = agents->get_warning[index];
	agent.alert_state = agents->alert_state[index];
	agent.sandbag_state = agents->sandbag_state[index];
	agent.sandbag_time_count = agents->sandbag_time_count[index];
	agent.flooded_time = agents->flooded_time[index];
	agent.initial_wl = agents->initial_wl[index];
	agent.actual_wl = agents->actual_wl[index];
	agent.average_wl = agents->average_wl[index];
	agent.max_wl = agents->max_wl[index];
	agent.financial_damage = agents->financial_damage[index];
	agent.inform_others = agents->inform_others[index];
	agent.get_informed = agents->get_informed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !output_financial_damage_infor(&agent, financial_damage_infor_messages	 );
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_financial_damage_infor Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->resident_num[index] = agent.resident_num;
	agents->OYI[index] = agent.OYI;
	agents->tenure[index] = agent.tenure;
	agents->take_measure[index] = agent.take_measure;
	agents->warning_area[index] = agent.warning_area;
	agents->get_warning[index] = agent.get_warning;
	agents->alert_state[index] = agent.alert_state;
	agents->sandbag_state[index] = agent.sandbag_state;
	agents->sandbag_time_count[index] = agent.sandbag_time_count;
	agents->flooded_time[index] = agent.flooded_time;
	agents->initial_wl[index] = agent.initial_wl;
	agents->actual_wl[index] = agent.actual_wl;
	agents->average_wl[index] = agent.average_wl;
	agents->max_wl[index] = agent.max_wl;
	agents->financial_damage[index] = agent.financial_damage;
	agents->inform_others[index] = agent.inform_others;
	agents->get_informed[index] = agent.get_informed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */


__global__ void GPUFLAME_identify_flood(xmachine_memory_household_list* agents, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_household_count)
        return;
    

	//SoA to AoS - xmachine_memory_identify_flood Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_household agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.resident_num = agents->resident_num[index];
	agent.OYI = agents->OYI[index];
	agent.tenure = agents->tenure[index];
	agent.take_measure = agents->take_measure[index];
	agent.warning_area = agents->warning_area[index];
	agent.get_warning = agents->get_warning[index];
	agent.alert_state = agents->alert_state[index];
	agent.sandbag_state = agents->sandbag_state[index];
	agent.sandbag_time_count = agents->sandbag_time_count[index];
	agent.flooded_time = agents->flooded_time[index];
	agent.initial_wl = agents->initial_wl[index];
	agent.actual_wl = agents->actual_wl[index];
	agent.average_wl = agents->average_wl[index];
	agent.max_wl = agents->max_wl[index];
	agent.financial_damage = agents->financial_damage[index];
	agent.inform_others = agents->inform_others[index];
	agent.get_informed = agents->get_informed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !identify_flood(&agent, flood_cell_messages, rand48 );
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_identify_flood Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->resident_num[index] = agent.resident_num;
	agents->OYI[index] = agent.OYI;
	agents->tenure[index] = agent.tenure;
	agents->take_measure[index] = agent.take_measure;
	agents->warning_area[index] = agent.warning_area;
	agents->get_warning[index] = agent.get_warning;
	agents->alert_state[index] = agent.alert_state;
	agents->sandbag_state[index] = agent.sandbag_state;
	agents->sandbag_time_count[index] = agent.sandbag_time_count;
	agents->flooded_time[index] = agent.flooded_time;
	agents->initial_wl[index] = agent.initial_wl;
	agents->actual_wl[index] = agent.actual_wl;
	agents->average_wl[index] = agent.average_wl;
	agents->max_wl[index] = agent.max_wl;
	agents->financial_damage[index] = agent.financial_damage;
	agents->inform_others[index] = agent.inform_others;
	agents->get_informed[index] = agent.get_informed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */


__global__ void GPUFLAME_detect_flood(xmachine_memory_household_list* agents, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_household_count)
        return;
    

	//SoA to AoS - xmachine_memory_detect_flood Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_household agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.resident_num = agents->resident_num[index];
	agent.OYI = agents->OYI[index];
	agent.tenure = agents->tenure[index];
	agent.take_measure = agents->take_measure[index];
	agent.warning_area = agents->warning_area[index];
	agent.get_warning = agents->get_warning[index];
	agent.alert_state = agents->alert_state[index];
	agent.sandbag_state = agents->sandbag_state[index];
	agent.sandbag_time_count = agents->sandbag_time_count[index];
	agent.flooded_time = agents->flooded_time[index];
	agent.initial_wl = agents->initial_wl[index];
	agent.actual_wl = agents->actual_wl[index];
	agent.average_wl = agents->average_wl[index];
	agent.max_wl = agents->max_wl[index];
	agent.financial_damage = agents->financial_damage[index];
	agent.inform_others = agents->inform_others[index];
	agent.get_informed = agents->get_informed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];

	//FLAME function call
	int dead = !detect_flood(&agent, flood_cell_messages, rand48 );
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_detect_flood Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->resident_num[index] = agent.resident_num;
	agents->OYI[index] = agent.OYI;
	agents->tenure[index] = agent.tenure;
	agents->take_measure[index] = agent.take_measure;
	agents->warning_area[index] = agent.warning_area;
	agents->get_warning[index] = agent.get_warning;
	agents->alert_state[index] = agent.alert_state;
	agents->sandbag_state[index] = agent.sandbag_state;
	agents->sandbag_time_count[index] = agent.sandbag_time_count;
	agents->flooded_time[index] = agent.flooded_time;
	agents->initial_wl[index] = agent.initial_wl;
	agents->actual_wl[index] = agent.actual_wl;
	agents->average_wl[index] = agent.average_wl;
	agents->max_wl[index] = agent.max_wl;
	agents->financial_damage[index] = agent.financial_damage;
	agents->inform_others[index] = agent.inform_others;
	agents->get_informed[index] = agent.get_informed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
}

/**
 *
 */


__global__ void GPUFLAME_communicate(xmachine_memory_household_list* agents, xmachine_message_state_data_list* state_data_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_communicate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_household agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_household_count){
    
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.resident_num = agents->resident_num[index];
	agent.OYI = agents->OYI[index];
	agent.tenure = agents->tenure[index];
	agent.take_measure = agents->take_measure[index];
	agent.warning_area = agents->warning_area[index];
	agent.get_warning = agents->get_warning[index];
	agent.alert_state = agents->alert_state[index];
	agent.sandbag_state = agents->sandbag_state[index];
	agent.sandbag_time_count = agents->sandbag_time_count[index];
	agent.flooded_time = agents->flooded_time[index];
	agent.initial_wl = agents->initial_wl[index];
	agent.actual_wl = agents->actual_wl[index];
	agent.average_wl = agents->average_wl[index];
	agent.max_wl = agents->max_wl[index];
	agent.financial_damage = agents->financial_damage[index];
	agent.inform_others = agents->inform_others[index];
	agent.get_informed = agents->get_informed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	} else {
	
	agent.x = 0;
	agent.y = 0;
	agent.resident_num = 0;
	agent.OYI = 0;
	agent.tenure = 0;
	agent.take_measure = 0;
	agent.warning_area = 0;
	agent.get_warning = 0;
	agent.alert_state = 0;
	agent.sandbag_state = 0;
	agent.sandbag_time_count = 0;
	agent.flooded_time = 0;
	agent.initial_wl = 0;
	agent.actual_wl = 0;
	agent.average_wl = 0;
	agent.max_wl = 0;
	agent.financial_damage = 0;
	agent.inform_others = 0;
	agent.get_informed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	}

	//FLAME function call
	int dead = !communicate(&agent, state_data_messages, rand48 );
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_household_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_communicate Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->resident_num[index] = agent.resident_num;
	agents->OYI[index] = agent.OYI;
	agents->tenure[index] = agent.tenure;
	agents->take_measure[index] = agent.take_measure;
	agents->warning_area[index] = agent.warning_area;
	agents->get_warning[index] = agent.get_warning;
	agents->alert_state[index] = agent.alert_state;
	agents->sandbag_state[index] = agent.sandbag_state;
	agents->sandbag_time_count[index] = agent.sandbag_time_count;
	agents->flooded_time[index] = agent.flooded_time;
	agents->initial_wl[index] = agent.initial_wl;
	agents->actual_wl[index] = agent.actual_wl;
	agents->average_wl[index] = agent.average_wl;
	agents->max_wl[index] = agent.max_wl;
	agents->financial_damage[index] = agent.financial_damage;
	agents->inform_others[index] = agent.inform_others;
	agents->get_informed[index] = agent.get_informed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	}
}

/**
 *
 */


__global__ void GPUFLAME_output_flood_cells(xmachine_memory_flood_list* agents, xmachine_message_flood_cell_list* flood_cell_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_output_flood_cells Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_flood agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.floodID = agents->floodID[index];
	agent.flood_h = agents->flood_h[index];

	//FLAME function call
	output_flood_cells(&agent, flood_cell_messages	 );
	

	

	//AoS to SoA - xmachine_memory_output_flood_cells Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->floodID[index] = agent.floodID;
	agents->flood_h[index] = agent.flood_h;
}

/**
 *
 */


__global__ void GPUFLAME_generate_warnings(xmachine_memory_flood_list* agents, xmachine_memory_warning_list* warning_agents){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_generate_warnings Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_flood agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.floodID = agents->floodID[index];
	agent.flood_h = agents->flood_h[index];

	//FLAME function call
	generate_warnings(&agent, warning_agents );
	

	

	//AoS to SoA - xmachine_memory_generate_warnings Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->floodID[index] = agent.floodID;
	agents->flood_h[index] = agent.flood_h;
}

/**
 *
 */


__global__ void GPUFLAME_update_data(xmachine_memory_flood_list* agents, short * data){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_update_data Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_flood agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.floodID = agents->floodID[index];
	agent.flood_h = agents->flood_h[index];

	//FLAME function call
	update_data(&agent, data );
	

	

	//AoS to SoA - xmachine_memory_update_data Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->floodID[index] = agent.floodID;
	agents->flood_h[index] = agent.flood_h;
}

/**
 *
 */


__global__ void GPUFLAME_calcu_damage_infor(xmachine_memory_warning_list* agents, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_calcu_damage_infor Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_warning agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_warning_count){
    
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.flooded_households = agents->flooded_households[index];
	agent.total_financial_damage = agents->total_financial_damage[index];
	agent.total_take_measure = agents->total_take_measure[index];
	agent.total_get_warning = agents->total_get_warning[index];
	agent.total_alert_state = agents->total_alert_state[index];
	agent.total_sandbag1 = agents->total_sandbag1[index];
	agent.total_sandbag2 = agents->total_sandbag2[index];
	agent.total_sandbag3 = agents->total_sandbag3[index];
	agent.total_inform_others = agents->total_inform_others[index];
	agent.total_get_informed = agents->total_get_informed[index];
	} else {
	
	agent.x = 0;
	agent.y = 0;
	agent.flooded_households = 0;
	agent.total_financial_damage = 0;
	agent.total_take_measure = 0;
	agent.total_get_warning = 0;
	agent.total_alert_state = 0;
	agent.total_sandbag1 = 0;
	agent.total_sandbag2 = 0;
	agent.total_sandbag3 = 0;
	agent.total_inform_others = 0;
	agent.total_get_informed = 0;
	}

	//FLAME function call
	int dead = !calcu_damage_infor(&agent, financial_damage_infor_messages );
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_warning_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_calcu_damage_infor Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->flooded_households[index] = agent.flooded_households;
	agents->total_financial_damage[index] = agent.total_financial_damage;
	agents->total_take_measure[index] = agent.total_take_measure;
	agents->total_get_warning[index] = agent.total_get_warning;
	agents->total_alert_state[index] = agent.total_alert_state;
	agents->total_sandbag1[index] = agent.total_sandbag1;
	agents->total_sandbag2[index] = agent.total_sandbag2;
	agents->total_sandbag3[index] = agent.total_sandbag3;
	agents->total_inform_others[index] = agent.total_inform_others;
	agents->total_get_informed[index] = agent.total_get_informed;
	}
}

/**
 *
 */


__global__ void GPUFLAME_output_state_data(xmachine_memory_warning_list* agents, xmachine_message_state_data_list* state_data_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_warning_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_state_data Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_warning agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.flooded_households = agents->flooded_households[index];
	agent.total_financial_damage = agents->total_financial_damage[index];
	agent.total_take_measure = agents->total_take_measure[index];
	agent.total_get_warning = agents->total_get_warning[index];
	agent.total_alert_state = agents->total_alert_state[index];
	agent.total_sandbag1 = agents->total_sandbag1[index];
	agent.total_sandbag2 = agents->total_sandbag2[index];
	agent.total_sandbag3 = agents->total_sandbag3[index];
	agent.total_inform_others = agents->total_inform_others[index];
	agent.total_get_informed = agents->total_get_informed[index];

	//FLAME function call
	int dead = !output_state_data(&agent, state_data_messages	 );
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_state_data Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->flooded_households[index] = agent.flooded_households;
	agents->total_financial_damage[index] = agent.total_financial_damage;
	agents->total_take_measure[index] = agent.total_take_measure;
	agents->total_get_warning[index] = agent.total_get_warning;
	agents->total_alert_state[index] = agent.total_alert_state;
	agents->total_sandbag1[index] = agent.total_sandbag1;
	agents->total_sandbag2[index] = agent.total_sandbag2;
	agents->total_sandbag3[index] = agent.total_sandbag3;
	agents->total_inform_others[index] = agent.total_inform_others;
	agents->total_get_informed[index] = agent.total_get_informed;
}

	
	
/* Graph utility functions */



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_
