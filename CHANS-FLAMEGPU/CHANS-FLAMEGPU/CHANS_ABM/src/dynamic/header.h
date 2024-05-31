
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



#ifndef __HEADER
#define __HEADER

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

	/* General standard definitions */
	//Threads per block (agents per block)
	#define THREADS_PER_TILE 64
	//Definition for any agent function or helper function
	#define __FLAME_GPU_FUNC__ __device__
	//Definition for a function used to initialise environment variables
	#define __FLAME_GPU_INIT_FUNC__
	#define __FLAME_GPU_STEP_FUNC__
	#define __FLAME_GPU_EXIT_FUNC__
	#define __FLAME_GPU_HOST_FUNC__ __host__

	#define USE_CUDA_STREAMS
	#define FAST_ATOMIC_SORTING

	// FLAME GPU Version Macros.
	#define FLAME_GPU_MAJOR_VERSION 1
	#define FLAME_GPU_MINOR_VERSION 5
	#define FLAME_GPU_PATCH_VERSION 0

	typedef unsigned int uint;

	//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
	typedef glm::vec2 fvec2;
	typedef glm::vec3 fvec3;
	typedef glm::vec4 fvec4;
	typedef glm::ivec2 ivec2;
	typedef glm::ivec3 ivec3;
	typedef glm::ivec4 ivec4;
	typedef glm::uvec2 uvec2;
	typedef glm::uvec3 uvec3;
	typedef glm::uvec4 uvec4;
	typedef glm::dvec2 dvec2;
	typedef glm::dvec3 dvec3;
	typedef glm::dvec4 dvec4;

	__device__ short* FLOOD_DATA_ARRAY2;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 1048576

//Maximum population size of xmachine_memory_household
#define xmachine_memory_household_MAX 1048576

//Maximum population size of xmachine_memory_flood
#define xmachine_memory_flood_MAX 1048576

//Maximum population size of xmachine_memory_warning
#define xmachine_memory_warning_MAX 1048576


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_flood_cell
#define xmachine_message_flood_cell_MAX 1048576

//Maximum population size of xmachine_mmessage_financial_damage_infor
#define xmachine_message_financial_damage_infor_MAX 1048576

//Maximum population size of xmachine_mmessage_state_data
#define xmachine_message_state_data_MAX 1048576


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_flood_cell_partitioningDiscrete
#define xmachine_message_financial_damage_infor_partitioningNone
#define xmachine_message_state_data_partitioningNone

/* Spatial partitioning grid size definitions */

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_household
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_household
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    int resident_num;    /**< X-machine memory variable resident_num of type int.*/
    int OYI;    /**< X-machine memory variable OYI of type int.*/
    int tenure;    /**< X-machine memory variable tenure of type int.*/
    int take_measure;    /**< X-machine memory variable take_measure of type int.*/
    int warning_area;    /**< X-machine memory variable warning_area of type int.*/
    int get_warning;    /**< X-machine memory variable get_warning of type int.*/
    int alert_state;    /**< X-machine memory variable alert_state of type int.*/
    int sandbag_state;    /**< X-machine memory variable sandbag_state of type int.*/
    int sandbag_time_count;    /**< X-machine memory variable sandbag_time_count of type int.*/
    int flooded_time;    /**< X-machine memory variable flooded_time of type int.*/
    float initial_wl;    /**< X-machine memory variable initial_wl of type float.*/
    float actual_wl;    /**< X-machine memory variable actual_wl of type float.*/
    float average_wl;    /**< X-machine memory variable average_wl of type float.*/
    float max_wl;    /**< X-machine memory variable max_wl of type float.*/
    float financial_damage;    /**< X-machine memory variable financial_damage of type float.*/
    int inform_others;    /**< X-machine memory variable inform_others of type int.*/
    int get_informed;    /**< X-machine memory variable get_informed of type int.*/
    int lod;    /**< X-machine memory variable lod of type int.*/
    float animate;    /**< X-machine memory variable animate of type float.*/
    int animate_dir;    /**< X-machine memory variable animate_dir of type int.*/
};

/** struct xmachine_memory_flood
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_flood
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    int floodID;    /**< X-machine memory variable floodID of type int.*/
    float flood_h;    /**< X-machine memory variable flood_h of type float.*/
};

/** struct xmachine_memory_warning
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_warning
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    int flooded_households;    /**< X-machine memory variable flooded_households of type int.*/
    float total_financial_damage;    /**< X-machine memory variable total_financial_damage of type float.*/
    int total_take_measure;    /**< X-machine memory variable total_take_measure of type int.*/
    int total_get_warning;    /**< X-machine memory variable total_get_warning of type int.*/
    int total_alert_state;    /**< X-machine memory variable total_alert_state of type int.*/
    int total_sandbag1;    /**< X-machine memory variable total_sandbag1 of type int.*/
    int total_sandbag2;    /**< X-machine memory variable total_sandbag2 of type int.*/
    int total_sandbag3;    /**< X-machine memory variable total_sandbag3 of type int.*/
    int total_inform_others;    /**< X-machine memory variable total_inform_others of type int.*/
    int total_get_informed;    /**< X-machine memory variable total_get_informed of type int.*/
};



/* Message structures */

/** struct xmachine_message_flood_cell
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_flood_cell
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    int floodID;        /**< Message variable floodID of type int.*/  
    float flood_h;        /**< Message variable flood_h of type float.*/
};

/** struct xmachine_message_financial_damage_infor
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_financial_damage_infor
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    float max_wl;        /**< Message variable max_wl of type float.*/  
    float financial_damage;        /**< Message variable financial_damage of type float.*/  
    int take_measure;        /**< Message variable take_measure of type int.*/  
    int get_warning;        /**< Message variable get_warning of type int.*/  
    int alert_state;        /**< Message variable alert_state of type int.*/  
    int sandbag_state;        /**< Message variable sandbag_state of type int.*/  
    int inform_others;        /**< Message variable inform_others of type int.*/  
    int get_informed;        /**< Message variable get_informed of type int.*/  
    int flooded_time;        /**< Message variable flooded_time of type int.*/  
    float actual_wl;        /**< Message variable actual_wl of type float.*/
};

/** struct xmachine_message_state_data
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_state_data
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    int flooded_households;        /**< Message variable flooded_households of type int.*/  
    float total_financial_damage;        /**< Message variable total_financial_damage of type float.*/  
    int total_take_measure;        /**< Message variable total_take_measure of type int.*/  
    int total_get_warning;        /**< Message variable total_get_warning of type int.*/  
    int total_alert_state;        /**< Message variable total_alert_state of type int.*/  
    int total_sandbag1;        /**< Message variable total_sandbag1 of type int.*/  
    int total_sandbag2;        /**< Message variable total_sandbag2 of type int.*/  
    int total_sandbag3;        /**< Message variable total_sandbag3 of type int.*/  
    int total_inform_others;        /**< Message variable total_inform_others of type int.*/  
    int total_get_informed;        /**< Message variable total_get_informed of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_household_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_household_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_household_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_household_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_household_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_household_MAX];    /**< X-machine memory variable list y of type float.*/
    int resident_num [xmachine_memory_household_MAX];    /**< X-machine memory variable list resident_num of type int.*/
    int OYI [xmachine_memory_household_MAX];    /**< X-machine memory variable list OYI of type int.*/
    int tenure [xmachine_memory_household_MAX];    /**< X-machine memory variable list tenure of type int.*/
    int take_measure [xmachine_memory_household_MAX];    /**< X-machine memory variable list take_measure of type int.*/
    int warning_area [xmachine_memory_household_MAX];    /**< X-machine memory variable list warning_area of type int.*/
    int get_warning [xmachine_memory_household_MAX];    /**< X-machine memory variable list get_warning of type int.*/
    int alert_state [xmachine_memory_household_MAX];    /**< X-machine memory variable list alert_state of type int.*/
    int sandbag_state [xmachine_memory_household_MAX];    /**< X-machine memory variable list sandbag_state of type int.*/
    int sandbag_time_count [xmachine_memory_household_MAX];    /**< X-machine memory variable list sandbag_time_count of type int.*/
    int flooded_time [xmachine_memory_household_MAX];    /**< X-machine memory variable list flooded_time of type int.*/
    float initial_wl [xmachine_memory_household_MAX];    /**< X-machine memory variable list initial_wl of type float.*/
    float actual_wl [xmachine_memory_household_MAX];    /**< X-machine memory variable list actual_wl of type float.*/
    float average_wl [xmachine_memory_household_MAX];    /**< X-machine memory variable list average_wl of type float.*/
    float max_wl [xmachine_memory_household_MAX];    /**< X-machine memory variable list max_wl of type float.*/
    float financial_damage [xmachine_memory_household_MAX];    /**< X-machine memory variable list financial_damage of type float.*/
    int inform_others [xmachine_memory_household_MAX];    /**< X-machine memory variable list inform_others of type int.*/
    int get_informed [xmachine_memory_household_MAX];    /**< X-machine memory variable list get_informed of type int.*/
    int lod [xmachine_memory_household_MAX];    /**< X-machine memory variable list lod of type int.*/
    float animate [xmachine_memory_household_MAX];    /**< X-machine memory variable list animate of type float.*/
    int animate_dir [xmachine_memory_household_MAX];    /**< X-machine memory variable list animate_dir of type int.*/
};

/** struct xmachine_memory_flood_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_flood_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_flood_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_flood_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_flood_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_flood_MAX];    /**< X-machine memory variable list y of type int.*/
    int floodID [xmachine_memory_flood_MAX];    /**< X-machine memory variable list floodID of type int.*/
    float flood_h [xmachine_memory_flood_MAX];    /**< X-machine memory variable list flood_h of type float.*/
};

/** struct xmachine_memory_warning_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_warning_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_warning_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_warning_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_warning_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_warning_MAX];    /**< X-machine memory variable list y of type float.*/
    int flooded_households [xmachine_memory_warning_MAX];    /**< X-machine memory variable list flooded_households of type int.*/
    float total_financial_damage [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_financial_damage of type float.*/
    int total_take_measure [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_take_measure of type int.*/
    int total_get_warning [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_get_warning of type int.*/
    int total_alert_state [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_alert_state of type int.*/
    int total_sandbag1 [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_sandbag1 of type int.*/
    int total_sandbag2 [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_sandbag2 of type int.*/
    int total_sandbag3 [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_sandbag3 of type int.*/
    int total_inform_others [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_inform_others of type int.*/
    int total_get_informed [xmachine_memory_warning_MAX];    /**< X-machine memory variable list total_get_informed of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_flood_cell_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_flood_cell_list
{
    int x [xmachine_message_flood_cell_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_flood_cell_MAX];    /**< Message memory variable list y of type int.*/
    int floodID [xmachine_message_flood_cell_MAX];    /**< Message memory variable list floodID of type int.*/
    float flood_h [xmachine_message_flood_cell_MAX];    /**< Message memory variable list flood_h of type float.*/
    
};

/** struct xmachine_message_financial_damage_infor_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_financial_damage_infor_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_financial_damage_infor_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_financial_damage_infor_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list z of type float.*/
    float max_wl [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list max_wl of type float.*/
    float financial_damage [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list financial_damage of type float.*/
    int take_measure [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list take_measure of type int.*/
    int get_warning [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list get_warning of type int.*/
    int alert_state [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list alert_state of type int.*/
    int sandbag_state [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list sandbag_state of type int.*/
    int inform_others [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list inform_others of type int.*/
    int get_informed [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list get_informed of type int.*/
    int flooded_time [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list flooded_time of type int.*/
    float actual_wl [xmachine_message_financial_damage_infor_MAX];    /**< Message memory variable list actual_wl of type float.*/
    
};

/** struct xmachine_message_state_data_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_state_data_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_state_data_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_state_data_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_state_data_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_state_data_MAX];    /**< Message memory variable list y of type float.*/
    int flooded_households [xmachine_message_state_data_MAX];    /**< Message memory variable list flooded_households of type int.*/
    float total_financial_damage [xmachine_message_state_data_MAX];    /**< Message memory variable list total_financial_damage of type float.*/
    int total_take_measure [xmachine_message_state_data_MAX];    /**< Message memory variable list total_take_measure of type int.*/
    int total_get_warning [xmachine_message_state_data_MAX];    /**< Message memory variable list total_get_warning of type int.*/
    int total_alert_state [xmachine_message_state_data_MAX];    /**< Message memory variable list total_alert_state of type int.*/
    int total_sandbag1 [xmachine_message_state_data_MAX];    /**< Message memory variable list total_sandbag1 of type int.*/
    int total_sandbag2 [xmachine_message_state_data_MAX];    /**< Message memory variable list total_sandbag2 of type int.*/
    int total_sandbag3 [xmachine_message_state_data_MAX];    /**< Message memory variable list total_sandbag3 of type int.*/
    int total_inform_others [xmachine_message_state_data_MAX];    /**< Message memory variable list total_inform_others of type int.*/
    int total_get_informed [xmachine_message_state_data_MAX];    /**< Message memory variable list total_get_informed of type int.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * output_financial_damage_infor FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_household. This represents a single agent instance and can be modified directly.
 * @param financial_damage_infor_messages Pointer to output message list of type xmachine_message_financial_damage_infor_list. Must be passed as an argument to the add_financial_damage_infor_message function ??.
 */
__FLAME_GPU_FUNC__ int output_financial_damage_infor(xmachine_memory_household* agent, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages);

/**
 * identify_flood FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_household. This represents a single agent instance and can be modified directly.
 * @param flood_cell_messages  flood_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_flood_cell_message and get_next_flood_cell_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int identify_flood(xmachine_memory_household* agent, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48);

/**
 * detect_flood FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_household. This represents a single agent instance and can be modified directly.
 * @param flood_cell_messages  flood_cell_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_flood_cell_message and get_next_flood_cell_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int detect_flood(xmachine_memory_household* agent, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48);

/**
 * communicate FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_household. This represents a single agent instance and can be modified directly.
 * @param state_data_messages  state_data_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_state_data_message and get_next_state_data_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int communicate(xmachine_memory_household* agent, xmachine_message_state_data_list* state_data_messages, RNG_rand48* rand48);

/**
 * output_flood_cells FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_flood. This represents a single agent instance and can be modified directly.
 * @param flood_cell_messages Pointer to output message list of type xmachine_message_flood_cell_list. Must be passed as an argument to the add_flood_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_flood_cells(xmachine_memory_flood* agent, xmachine_message_flood_cell_list* flood_cell_messages);

/**
 * generate_warnings FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_flood. This represents a single agent instance and can be modified directly.
 * @param warning_agents Pointer to agent list of type xmachine_memory_warning_list. This must be passed as an argument to the add_warning_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int generate_warnings(xmachine_memory_flood* agent, xmachine_memory_warning_list* warning_agents);

/**
 * update_data FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_flood. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int update_data(xmachine_memory_flood* agent);

/**
 * calcu_damage_infor FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_warning. This represents a single agent instance and can be modified directly.
 * @param financial_damage_infor_messages  financial_damage_infor_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_financial_damage_infor_message and get_next_financial_damage_infor_message functions.
 */
__FLAME_GPU_FUNC__ int calcu_damage_infor(xmachine_memory_warning* agent, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages);

/**
 * output_state_data FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_warning. This represents a single agent instance and can be modified directly.
 * @param state_data_messages Pointer to output message list of type xmachine_message_state_data_list. Must be passed as an argument to the add_state_data_message function ??.
 */
__FLAME_GPU_FUNC__ int output_state_data(xmachine_memory_warning* agent, xmachine_message_state_data_list* state_data_messages);

  
/* Message Function Prototypes for Discrete Partitioned flood_cell message implemented in FLAMEGPU_Kernels */

/** add_flood_cell_message
 * Function for all types of message partitioning
 * Adds a new flood_cell agent to the xmachine_memory_flood_cell_list list using a linear mapping
 * @param agents	xmachine_memory_flood_cell_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param floodID	message variable of type int
 * @param flood_h	message variable of type float
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_flood_cell_message(xmachine_message_flood_cell_list* flood_cell_messages, int x, int y, int floodID, float flood_h);
 
/** get_first_flood_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param flood_cell_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_flood_cell * get_first_flood_cell_message(xmachine_message_flood_cell_list* flood_cell_messages, int agentx, int agent_y);

/** get_next_flood_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param flood_cell_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_flood_cell * get_next_flood_cell_message(xmachine_message_flood_cell* current, xmachine_message_flood_cell_list* flood_cell_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) financial_damage_infor message implemented in FLAMEGPU_Kernels */

/** add_financial_damage_infor_message
 * Function for all types of message partitioning
 * Adds a new financial_damage_infor agent to the xmachine_memory_financial_damage_infor_list list using a linear mapping
 * @param agents	xmachine_memory_financial_damage_infor_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param max_wl	message variable of type float
 * @param financial_damage	message variable of type float
 * @param take_measure	message variable of type int
 * @param get_warning	message variable of type int
 * @param alert_state	message variable of type int
 * @param sandbag_state	message variable of type int
 * @param inform_others	message variable of type int
 * @param get_informed	message variable of type int
 * @param flooded_time	message variable of type int
 * @param actual_wl	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_financial_damage_infor_message(xmachine_message_financial_damage_infor_list* financial_damage_infor_messages, float x, float y, float z, float max_wl, float financial_damage, int take_measure, int get_warning, int alert_state, int sandbag_state, int inform_others, int get_informed, int flooded_time, float actual_wl);
 
/** get_first_financial_damage_infor_message
 * Get first message function for non partitioned (brute force) messages
 * @param financial_damage_infor_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_financial_damage_infor * get_first_financial_damage_infor_message(xmachine_message_financial_damage_infor_list* financial_damage_infor_messages);

/** get_next_financial_damage_infor_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param financial_damage_infor_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_financial_damage_infor * get_next_financial_damage_infor_message(xmachine_message_financial_damage_infor* current, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) state_data message implemented in FLAMEGPU_Kernels */

/** add_state_data_message
 * Function for all types of message partitioning
 * Adds a new state_data agent to the xmachine_memory_state_data_list list using a linear mapping
 * @param agents	xmachine_memory_state_data_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param flooded_households	message variable of type int
 * @param total_financial_damage	message variable of type float
 * @param total_take_measure	message variable of type int
 * @param total_get_warning	message variable of type int
 * @param total_alert_state	message variable of type int
 * @param total_sandbag1	message variable of type int
 * @param total_sandbag2	message variable of type int
 * @param total_sandbag3	message variable of type int
 * @param total_inform_others	message variable of type int
 * @param total_get_informed	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_state_data_message(xmachine_message_state_data_list* state_data_messages, float x, float y, int flooded_households, float total_financial_damage, int total_take_measure, int total_get_warning, int total_alert_state, int total_sandbag1, int total_sandbag2, int total_sandbag3, int total_inform_others, int total_get_informed);
 
/** get_first_state_data_message
 * Get first message function for non partitioned (brute force) messages
 * @param state_data_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_state_data * get_first_state_data_message(xmachine_message_state_data_list* state_data_messages);

/** get_next_state_data_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param state_data_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_state_data * get_next_state_data_message(xmachine_message_state_data* current, xmachine_message_state_data_list* state_data_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_household_agent
 * Adds a new continuous valued household agent to the xmachine_memory_household_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_household_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param resident_num	agent agent variable of type int
 * @param OYI	agent agent variable of type int
 * @param tenure	agent agent variable of type int
 * @param take_measure	agent agent variable of type int
 * @param warning_area	agent agent variable of type int
 * @param get_warning	agent agent variable of type int
 * @param alert_state	agent agent variable of type int
 * @param sandbag_state	agent agent variable of type int
 * @param sandbag_time_count	agent agent variable of type int
 * @param flooded_time	agent agent variable of type int
 * @param initial_wl	agent agent variable of type float
 * @param actual_wl	agent agent variable of type float
 * @param average_wl	agent agent variable of type float
 * @param max_wl	agent agent variable of type float
 * @param financial_damage	agent agent variable of type float
 * @param inform_others	agent agent variable of type int
 * @param get_informed	agent agent variable of type int
 * @param lod	agent agent variable of type int
 * @param animate	agent agent variable of type float
 * @param animate_dir	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_household_agent(xmachine_memory_household_list* agents, float x, float y, int resident_num, int OYI, int tenure, int take_measure, int warning_area, int get_warning, int alert_state, int sandbag_state, int sandbag_time_count, int flooded_time, float initial_wl, float actual_wl, float average_wl, float max_wl, float financial_damage, int inform_others, int get_informed, int lod, float animate, int animate_dir);

/** add_warning_agent
 * Adds a new continuous valued warning agent to the xmachine_memory_warning_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_warning_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param flooded_households	agent agent variable of type int
 * @param total_financial_damage	agent agent variable of type float
 * @param total_take_measure	agent agent variable of type int
 * @param total_get_warning	agent agent variable of type int
 * @param total_alert_state	agent agent variable of type int
 * @param total_sandbag1	agent agent variable of type int
 * @param total_sandbag2	agent agent variable of type int
 * @param total_sandbag3	agent agent variable of type int
 * @param total_inform_others	agent agent variable of type int
 * @param total_get_informed	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_warning_agent(xmachine_memory_warning_list* agents, float x, float y, int flooded_households, float total_financial_damage, int total_take_measure, int total_get_warning, int total_alert_state, int total_sandbag1, int total_sandbag2, int total_sandbag3, int total_inform_others, int total_get_informed);


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_households Pointer to agent list on the host
 * @param d_households Pointer to agent list on the GPU device
 * @param h_xmachine_memory_household_count Pointer to agent counter
 * @param h_floods Pointer to agent list on the host
 * @param d_floods Pointer to agent list on the GPU device
 * @param h_xmachine_memory_flood_count Pointer to agent counter
 * @param h_warnings Pointer to agent list on the host
 * @param d_warnings Pointer to agent list on the GPU device
 * @param h_xmachine_memory_warning_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_household_list* h_households_default, xmachine_memory_household_list* d_households_default, int h_xmachine_memory_household_default_count,xmachine_memory_flood_list* h_floods_static, xmachine_memory_flood_list* d_floods_static, int h_xmachine_memory_flood_static_count,xmachine_memory_warning_list* h_warnings_static_warning, xmachine_memory_warning_list* d_warnings_static_warning, int h_xmachine_memory_warning_static_warning_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_households Pointer to agent list on the host
 * @param h_xmachine_memory_household_count Pointer to agent counter
 * @param h_floods Pointer to agent list on the host
 * @param h_xmachine_memory_flood_count Pointer to agent counter
 * @param h_warnings Pointer to agent list on the host
 * @param h_xmachine_memory_warning_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_household_list* h_households, int* h_xmachine_memory_household_count,xmachine_memory_flood_list* h_floods, int* h_xmachine_memory_flood_count,xmachine_memory_warning_list* h_warnings, int* h_xmachine_memory_warning_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_household_MAX_count
 * Gets the max agent count for the household agent type 
 * @return		the maximum household agent count
 */
extern int get_agent_household_MAX_count();



/** get_agent_household_default_count
 * Gets the agent count for the household agent type in state default
 * @return		the current household agent count in state default
 */
extern int get_agent_household_default_count();

/** reset_default_count
 * Resets the agent count of the household in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_household_default_count();

/** get_device_household_default_agents
 * Gets a pointer to xmachine_memory_household_list on the GPU device
 * @return		a xmachine_memory_household_list on the GPU device
 */
extern xmachine_memory_household_list* get_device_household_default_agents();

/** get_host_household_default_agents
 * Gets a pointer to xmachine_memory_household_list on the CPU host
 * @return		a xmachine_memory_household_list on the CPU host
 */
extern xmachine_memory_household_list* get_host_household_default_agents();


/** sort_households_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_households_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_household_list* agents));


    
/** get_agent_flood_MAX_count
 * Gets the max agent count for the flood agent type 
 * @return		the maximum flood agent count
 */
extern int get_agent_flood_MAX_count();



/** get_agent_flood_static_count
 * Gets the agent count for the flood agent type in state static
 * @return		the current flood agent count in state static
 */
extern int get_agent_flood_static_count();

/** reset_static_count
 * Resets the agent count of the flood in state static to 0. This is useful for interacting with some visualisations.
 */
extern void reset_flood_static_count();

/** get_device_flood_static_agents
 * Gets a pointer to xmachine_memory_flood_list on the GPU device
 * @return		a xmachine_memory_flood_list on the GPU device
 */
extern xmachine_memory_flood_list* get_device_flood_static_agents();

/** get_host_flood_static_agents
 * Gets a pointer to xmachine_memory_flood_list on the CPU host
 * @return		a xmachine_memory_flood_list on the CPU host
 */
extern xmachine_memory_flood_list* get_host_flood_static_agents();


/** get_flood_population_width
 * Gets an int value representing the xmachine_memory_flood population width.
 * @return		xmachine_memory_flood population width
 */
extern int get_flood_population_width();

    
/** get_agent_warning_MAX_count
 * Gets the max agent count for the warning agent type 
 * @return		the maximum warning agent count
 */
extern int get_agent_warning_MAX_count();



/** get_agent_warning_static_warning_count
 * Gets the agent count for the warning agent type in state static_warning
 * @return		the current warning agent count in state static_warning
 */
extern int get_agent_warning_static_warning_count();

/** reset_static_warning_count
 * Resets the agent count of the warning in state static_warning to 0. This is useful for interacting with some visualisations.
 */
extern void reset_warning_static_warning_count();

/** get_device_warning_static_warning_agents
 * Gets a pointer to xmachine_memory_warning_list on the GPU device
 * @return		a xmachine_memory_warning_list on the GPU device
 */
extern xmachine_memory_warning_list* get_device_warning_static_warning_agents();

/** get_host_warning_static_warning_agents
 * Gets a pointer to xmachine_memory_warning_list on the CPU host
 * @return		a xmachine_memory_warning_list on the CPU host
 */
extern xmachine_memory_warning_list* get_host_warning_static_warning_agents();


/** sort_warnings_static_warning
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_warnings_static_warning(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_warning_list* agents));



/* Host based access of agent variables*/

/** float get_household_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_household_default_variable_x(unsigned int index);

/** float get_household_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_household_default_variable_y(unsigned int index);

/** int get_household_default_variable_resident_num(unsigned int index)
 * Gets the value of the resident_num variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable resident_num
 */
__host__ int get_household_default_variable_resident_num(unsigned int index);

/** int get_household_default_variable_OYI(unsigned int index)
 * Gets the value of the OYI variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable OYI
 */
__host__ int get_household_default_variable_OYI(unsigned int index);

/** int get_household_default_variable_tenure(unsigned int index)
 * Gets the value of the tenure variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tenure
 */
__host__ int get_household_default_variable_tenure(unsigned int index);

/** int get_household_default_variable_take_measure(unsigned int index)
 * Gets the value of the take_measure variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable take_measure
 */
__host__ int get_household_default_variable_take_measure(unsigned int index);

/** int get_household_default_variable_warning_area(unsigned int index)
 * Gets the value of the warning_area variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable warning_area
 */
__host__ int get_household_default_variable_warning_area(unsigned int index);

/** int get_household_default_variable_get_warning(unsigned int index)
 * Gets the value of the get_warning variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable get_warning
 */
__host__ int get_household_default_variable_get_warning(unsigned int index);

/** int get_household_default_variable_alert_state(unsigned int index)
 * Gets the value of the alert_state variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable alert_state
 */
__host__ int get_household_default_variable_alert_state(unsigned int index);

/** int get_household_default_variable_sandbag_state(unsigned int index)
 * Gets the value of the sandbag_state variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sandbag_state
 */
__host__ int get_household_default_variable_sandbag_state(unsigned int index);

/** int get_household_default_variable_sandbag_time_count(unsigned int index)
 * Gets the value of the sandbag_time_count variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sandbag_time_count
 */
__host__ int get_household_default_variable_sandbag_time_count(unsigned int index);

/** int get_household_default_variable_flooded_time(unsigned int index)
 * Gets the value of the flooded_time variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flooded_time
 */
__host__ int get_household_default_variable_flooded_time(unsigned int index);

/** float get_household_default_variable_initial_wl(unsigned int index)
 * Gets the value of the initial_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable initial_wl
 */
__host__ float get_household_default_variable_initial_wl(unsigned int index);

/** float get_household_default_variable_actual_wl(unsigned int index)
 * Gets the value of the actual_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable actual_wl
 */
__host__ float get_household_default_variable_actual_wl(unsigned int index);

/** float get_household_default_variable_average_wl(unsigned int index)
 * Gets the value of the average_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable average_wl
 */
__host__ float get_household_default_variable_average_wl(unsigned int index);

/** float get_household_default_variable_max_wl(unsigned int index)
 * Gets the value of the max_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable max_wl
 */
__host__ float get_household_default_variable_max_wl(unsigned int index);

/** float get_household_default_variable_financial_damage(unsigned int index)
 * Gets the value of the financial_damage variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable financial_damage
 */
__host__ float get_household_default_variable_financial_damage(unsigned int index);

/** int get_household_default_variable_inform_others(unsigned int index)
 * Gets the value of the inform_others variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable inform_others
 */
__host__ int get_household_default_variable_inform_others(unsigned int index);

/** int get_household_default_variable_get_informed(unsigned int index)
 * Gets the value of the get_informed variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable get_informed
 */
__host__ int get_household_default_variable_get_informed(unsigned int index);

/** int get_household_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_household_default_variable_lod(unsigned int index);

/** float get_household_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_household_default_variable_animate(unsigned int index);

/** int get_household_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_household_default_variable_animate_dir(unsigned int index);

/** int get_flood_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_flood_static_variable_x(unsigned int index);

/** int get_flood_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_flood_static_variable_y(unsigned int index);

/** int get_flood_static_variable_floodID(unsigned int index)
 * Gets the value of the floodID variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable floodID
 */
__host__ int get_flood_static_variable_floodID(unsigned int index);

/** float get_flood_static_variable_flood_h(unsigned int index)
 * Gets the value of the flood_h variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flood_h
 */
__host__ float get_flood_static_variable_flood_h(unsigned int index);

/** float get_warning_static_warning_variable_x(unsigned int index)
 * Gets the value of the x variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_warning_static_warning_variable_x(unsigned int index);

/** float get_warning_static_warning_variable_y(unsigned int index)
 * Gets the value of the y variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_warning_static_warning_variable_y(unsigned int index);

/** int get_warning_static_warning_variable_flooded_households(unsigned int index)
 * Gets the value of the flooded_households variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flooded_households
 */
__host__ int get_warning_static_warning_variable_flooded_households(unsigned int index);

/** float get_warning_static_warning_variable_total_financial_damage(unsigned int index)
 * Gets the value of the total_financial_damage variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_financial_damage
 */
__host__ float get_warning_static_warning_variable_total_financial_damage(unsigned int index);

/** int get_warning_static_warning_variable_total_take_measure(unsigned int index)
 * Gets the value of the total_take_measure variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_take_measure
 */
__host__ int get_warning_static_warning_variable_total_take_measure(unsigned int index);

/** int get_warning_static_warning_variable_total_get_warning(unsigned int index)
 * Gets the value of the total_get_warning variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_get_warning
 */
__host__ int get_warning_static_warning_variable_total_get_warning(unsigned int index);

/** int get_warning_static_warning_variable_total_alert_state(unsigned int index)
 * Gets the value of the total_alert_state variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_alert_state
 */
__host__ int get_warning_static_warning_variable_total_alert_state(unsigned int index);

/** int get_warning_static_warning_variable_total_sandbag1(unsigned int index)
 * Gets the value of the total_sandbag1 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag1
 */
__host__ int get_warning_static_warning_variable_total_sandbag1(unsigned int index);

/** int get_warning_static_warning_variable_total_sandbag2(unsigned int index)
 * Gets the value of the total_sandbag2 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag2
 */
__host__ int get_warning_static_warning_variable_total_sandbag2(unsigned int index);

/** int get_warning_static_warning_variable_total_sandbag3(unsigned int index)
 * Gets the value of the total_sandbag3 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag3
 */
__host__ int get_warning_static_warning_variable_total_sandbag3(unsigned int index);

/** int get_warning_static_warning_variable_total_inform_others(unsigned int index)
 * Gets the value of the total_inform_others variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_inform_others
 */
__host__ int get_warning_static_warning_variable_total_inform_others(unsigned int index);

/** int get_warning_static_warning_variable_total_get_informed(unsigned int index)
 * Gets the value of the total_get_informed variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_get_informed
 */
__host__ int get_warning_static_warning_variable_total_get_informed(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_household
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated household struct.
 */
xmachine_memory_household* h_allocate_agent_household();
/** h_free_agent_household
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_household(xmachine_memory_household** agent);
/** h_allocate_agent_household_array
 * Utility function to allocate an array of structs for  household agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_household** h_allocate_agent_household_array(unsigned int count);
/** h_free_agent_household_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_household_array(xmachine_memory_household*** agents, unsigned int count);


/** h_add_agent_household_default
 * Host function to add a single agent of type household to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_household_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_household_default(xmachine_memory_household* agent);

/** h_add_agents_household_default(
 * Host function to add multiple agents of type household to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of household agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_household_default(xmachine_memory_household** agents, unsigned int count);

/** h_allocate_agent_flood
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated flood struct.
 */
xmachine_memory_flood* h_allocate_agent_flood();
/** h_free_agent_flood
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_flood(xmachine_memory_flood** agent);
/** h_allocate_agent_flood_array
 * Utility function to allocate an array of structs for  flood agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_flood** h_allocate_agent_flood_array(unsigned int count);
/** h_free_agent_flood_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_flood_array(xmachine_memory_flood*** agents, unsigned int count);


/** h_add_agent_flood_static
 * Host function to add a single agent of type flood to the static state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_flood_static instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_flood_static(xmachine_memory_flood* agent);

/** h_add_agents_flood_static(
 * Host function to add multiple agents of type flood to the static state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of flood agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_flood_static(xmachine_memory_flood** agents, unsigned int count);

/** h_allocate_agent_warning
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated warning struct.
 */
xmachine_memory_warning* h_allocate_agent_warning();
/** h_free_agent_warning
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_warning(xmachine_memory_warning** agent);
/** h_allocate_agent_warning_array
 * Utility function to allocate an array of structs for  warning agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_warning** h_allocate_agent_warning_array(unsigned int count);
/** h_free_agent_warning_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_warning_array(xmachine_memory_warning*** agents, unsigned int count);


/** h_add_agent_warning_static_warning
 * Host function to add a single agent of type warning to the static_warning state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_warning_static_warning instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_warning_static_warning(xmachine_memory_warning* agent);

/** h_add_agents_warning_static_warning(
 * Host function to add multiple agents of type warning to the static_warning state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of warning agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_warning_static_warning(xmachine_memory_warning** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** float reduce_household_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_x_variable();



/** float min_household_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_x_variable();
/** float max_household_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_x_variable();

/** float reduce_household_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_y_variable();



/** float min_household_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_y_variable();
/** float max_household_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_y_variable();

/** int reduce_household_default_resident_num_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_resident_num_variable();



/** int count_household_default_resident_num_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_resident_num_variable(int count_value);

/** int min_household_default_resident_num_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_resident_num_variable();
/** int max_household_default_resident_num_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_resident_num_variable();

/** int reduce_household_default_OYI_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_OYI_variable();



/** int count_household_default_OYI_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_OYI_variable(int count_value);

/** int min_household_default_OYI_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_OYI_variable();
/** int max_household_default_OYI_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_OYI_variable();

/** int reduce_household_default_tenure_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_tenure_variable();



/** int count_household_default_tenure_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_tenure_variable(int count_value);

/** int min_household_default_tenure_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_tenure_variable();
/** int max_household_default_tenure_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_tenure_variable();

/** int reduce_household_default_take_measure_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_take_measure_variable();



/** int count_household_default_take_measure_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_take_measure_variable(int count_value);

/** int min_household_default_take_measure_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_take_measure_variable();
/** int max_household_default_take_measure_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_take_measure_variable();

/** int reduce_household_default_warning_area_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_warning_area_variable();



/** int count_household_default_warning_area_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_warning_area_variable(int count_value);

/** int min_household_default_warning_area_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_warning_area_variable();
/** int max_household_default_warning_area_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_warning_area_variable();

/** int reduce_household_default_get_warning_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_get_warning_variable();



/** int count_household_default_get_warning_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_get_warning_variable(int count_value);

/** int min_household_default_get_warning_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_get_warning_variable();
/** int max_household_default_get_warning_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_get_warning_variable();

/** int reduce_household_default_alert_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_alert_state_variable();



/** int count_household_default_alert_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_alert_state_variable(int count_value);

/** int min_household_default_alert_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_alert_state_variable();
/** int max_household_default_alert_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_alert_state_variable();

/** int reduce_household_default_sandbag_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_sandbag_state_variable();



/** int count_household_default_sandbag_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_sandbag_state_variable(int count_value);

/** int min_household_default_sandbag_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_sandbag_state_variable();
/** int max_household_default_sandbag_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_sandbag_state_variable();

/** int reduce_household_default_sandbag_time_count_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_sandbag_time_count_variable();



/** int count_household_default_sandbag_time_count_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_sandbag_time_count_variable(int count_value);

/** int min_household_default_sandbag_time_count_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_sandbag_time_count_variable();
/** int max_household_default_sandbag_time_count_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_sandbag_time_count_variable();

/** int reduce_household_default_flooded_time_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_flooded_time_variable();



/** int count_household_default_flooded_time_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_flooded_time_variable(int count_value);

/** int min_household_default_flooded_time_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_flooded_time_variable();
/** int max_household_default_flooded_time_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_flooded_time_variable();

/** float reduce_household_default_initial_wl_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_initial_wl_variable();



/** float min_household_default_initial_wl_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_initial_wl_variable();
/** float max_household_default_initial_wl_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_initial_wl_variable();

/** float reduce_household_default_actual_wl_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_actual_wl_variable();



/** float min_household_default_actual_wl_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_actual_wl_variable();
/** float max_household_default_actual_wl_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_actual_wl_variable();

/** float reduce_household_default_average_wl_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_average_wl_variable();



/** float min_household_default_average_wl_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_average_wl_variable();
/** float max_household_default_average_wl_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_average_wl_variable();

/** float reduce_household_default_max_wl_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_max_wl_variable();



/** float min_household_default_max_wl_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_max_wl_variable();
/** float max_household_default_max_wl_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_max_wl_variable();

/** float reduce_household_default_financial_damage_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_financial_damage_variable();



/** float min_household_default_financial_damage_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_financial_damage_variable();
/** float max_household_default_financial_damage_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_financial_damage_variable();

/** int reduce_household_default_inform_others_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_inform_others_variable();



/** int count_household_default_inform_others_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_inform_others_variable(int count_value);

/** int min_household_default_inform_others_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_inform_others_variable();
/** int max_household_default_inform_others_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_inform_others_variable();

/** int reduce_household_default_get_informed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_get_informed_variable();



/** int count_household_default_get_informed_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_get_informed_variable(int count_value);

/** int min_household_default_get_informed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_get_informed_variable();
/** int max_household_default_get_informed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_get_informed_variable();

/** int reduce_household_default_lod_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_lod_variable();



/** int count_household_default_lod_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_lod_variable(int count_value);

/** int min_household_default_lod_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_lod_variable();
/** int max_household_default_lod_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_lod_variable();

/** float reduce_household_default_animate_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_household_default_animate_variable();



/** float min_household_default_animate_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_household_default_animate_variable();
/** float max_household_default_animate_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_household_default_animate_variable();

/** int reduce_household_default_animate_dir_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_household_default_animate_dir_variable();



/** int count_household_default_animate_dir_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_household_default_animate_dir_variable(int count_value);

/** int min_household_default_animate_dir_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_household_default_animate_dir_variable();
/** int max_household_default_animate_dir_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_household_default_animate_dir_variable();

/** int reduce_flood_static_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_flood_static_x_variable();



/** int count_flood_static_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_flood_static_x_variable(int count_value);

/** int min_flood_static_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_flood_static_x_variable();
/** int max_flood_static_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_flood_static_x_variable();

/** int reduce_flood_static_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_flood_static_y_variable();



/** int count_flood_static_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_flood_static_y_variable(int count_value);

/** int min_flood_static_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_flood_static_y_variable();
/** int max_flood_static_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_flood_static_y_variable();

/** int reduce_flood_static_floodID_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_flood_static_floodID_variable();



/** int count_flood_static_floodID_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_flood_static_floodID_variable(int count_value);

/** int min_flood_static_floodID_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_flood_static_floodID_variable();
/** int max_flood_static_floodID_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_flood_static_floodID_variable();

/** float reduce_flood_static_flood_h_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_flood_static_flood_h_variable();



/** float min_flood_static_flood_h_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_flood_static_flood_h_variable();
/** float max_flood_static_flood_h_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_flood_static_flood_h_variable();

/** float reduce_warning_static_warning_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_warning_static_warning_x_variable();



/** float min_warning_static_warning_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_warning_static_warning_x_variable();
/** float max_warning_static_warning_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_warning_static_warning_x_variable();

/** float reduce_warning_static_warning_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_warning_static_warning_y_variable();



/** float min_warning_static_warning_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_warning_static_warning_y_variable();
/** float max_warning_static_warning_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_warning_static_warning_y_variable();

/** int reduce_warning_static_warning_flooded_households_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_flooded_households_variable();



/** int count_warning_static_warning_flooded_households_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_flooded_households_variable(int count_value);

/** int min_warning_static_warning_flooded_households_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_flooded_households_variable();
/** int max_warning_static_warning_flooded_households_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_flooded_households_variable();

/** float reduce_warning_static_warning_total_financial_damage_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_warning_static_warning_total_financial_damage_variable();



/** float min_warning_static_warning_total_financial_damage_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_warning_static_warning_total_financial_damage_variable();
/** float max_warning_static_warning_total_financial_damage_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_warning_static_warning_total_financial_damage_variable();

/** int reduce_warning_static_warning_total_take_measure_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_take_measure_variable();



/** int count_warning_static_warning_total_take_measure_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_take_measure_variable(int count_value);

/** int min_warning_static_warning_total_take_measure_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_take_measure_variable();
/** int max_warning_static_warning_total_take_measure_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_take_measure_variable();

/** int reduce_warning_static_warning_total_get_warning_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_get_warning_variable();



/** int count_warning_static_warning_total_get_warning_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_get_warning_variable(int count_value);

/** int min_warning_static_warning_total_get_warning_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_get_warning_variable();
/** int max_warning_static_warning_total_get_warning_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_get_warning_variable();

/** int reduce_warning_static_warning_total_alert_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_alert_state_variable();



/** int count_warning_static_warning_total_alert_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_alert_state_variable(int count_value);

/** int min_warning_static_warning_total_alert_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_alert_state_variable();
/** int max_warning_static_warning_total_alert_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_alert_state_variable();

/** int reduce_warning_static_warning_total_sandbag1_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_sandbag1_variable();



/** int count_warning_static_warning_total_sandbag1_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_sandbag1_variable(int count_value);

/** int min_warning_static_warning_total_sandbag1_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_sandbag1_variable();
/** int max_warning_static_warning_total_sandbag1_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_sandbag1_variable();

/** int reduce_warning_static_warning_total_sandbag2_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_sandbag2_variable();



/** int count_warning_static_warning_total_sandbag2_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_sandbag2_variable(int count_value);

/** int min_warning_static_warning_total_sandbag2_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_sandbag2_variable();
/** int max_warning_static_warning_total_sandbag2_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_sandbag2_variable();

/** int reduce_warning_static_warning_total_sandbag3_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_sandbag3_variable();



/** int count_warning_static_warning_total_sandbag3_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_sandbag3_variable(int count_value);

/** int min_warning_static_warning_total_sandbag3_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_sandbag3_variable();
/** int max_warning_static_warning_total_sandbag3_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_sandbag3_variable();

/** int reduce_warning_static_warning_total_inform_others_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_inform_others_variable();



/** int count_warning_static_warning_total_inform_others_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_inform_others_variable(int count_value);

/** int min_warning_static_warning_total_inform_others_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_inform_others_variable();
/** int max_warning_static_warning_total_inform_others_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_inform_others_variable();

/** int reduce_warning_static_warning_total_get_informed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_warning_static_warning_total_get_informed_variable();



/** int count_warning_static_warning_total_get_informed_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_warning_static_warning_total_get_informed_variable(int count_value);

/** int min_warning_static_warning_total_get_informed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_warning_static_warning_total_get_informed_variable();
/** int max_warning_static_warning_total_get_informed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_warning_static_warning_total_get_informed_variable();


  
/* global constant variables */

__device__ short * FLOOD_DATA_ARRAY;
			
__constant__ int TIME;
			
__constant__ int RANDOM_SEED_SEC;
			
__constant__ int RANDOM_SEED_MIN;
			
__constant__ float TIME_SCALER;
			
/** set_FLOOD_DATA_ARRAY
 * Sets the constant variable FLOOD_DATA_ARRAY on the device which can then be used in the agent functions.
 * @param h_FLOOD_DATA_ARRAY value to set the variable
 */
extern void set_FLOOD_DATA_ARRAY(short* h_FLOOD_DATA_ARRAY);

extern const short* get_FLOOD_DATA_ARRAY();


extern short h_env_FLOOD_DATA_ARRAY[900000];

/** set_TIME
 * Sets the constant variable TIME on the device which can then be used in the agent functions.
 * @param h_TIME value to set the variable
 */
extern void set_TIME(int* h_TIME);

extern const int* get_TIME();


extern int h_env_TIME;

/** set_RANDOM_SEED_SEC
 * Sets the constant variable RANDOM_SEED_SEC on the device which can then be used in the agent functions.
 * @param h_RANDOM_SEED_SEC value to set the variable
 */
extern void set_RANDOM_SEED_SEC(int* h_RANDOM_SEED_SEC);

extern const int* get_RANDOM_SEED_SEC();


extern int h_env_RANDOM_SEED_SEC;

/** set_RANDOM_SEED_MIN
 * Sets the constant variable RANDOM_SEED_MIN on the device which can then be used in the agent functions.
 * @param h_RANDOM_SEED_MIN value to set the variable
 */
extern void set_RANDOM_SEED_MIN(int* h_RANDOM_SEED_MIN);

extern const int* get_RANDOM_SEED_MIN();


extern int h_env_RANDOM_SEED_MIN;

/** set_TIME_SCALER
 * Sets the constant variable TIME_SCALER on the device which can then be used in the agent functions.
 * @param h_TIME_SCALER value to set the variable
 */
extern void set_TIME_SCALER(float* h_TIME_SCALER);

extern const float* get_TIME_SCALER();


extern float h_env_TIME_SCALER;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

