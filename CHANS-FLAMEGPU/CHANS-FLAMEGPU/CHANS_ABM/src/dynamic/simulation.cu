
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


  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* household Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_household_list* d_households;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_household_list* d_households_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_household_list* d_households_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_household_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_household_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_household_values;  /**< Agent sort identifiers value */

/* household state variables */
xmachine_memory_household_list* h_households_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_household_list* d_households_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_household_default_count;   /**< Agent population size counter */ 

/* flood Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_flood_list* d_floods;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_flood_list* d_floods_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_flood_list* d_floods_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_flood_count;   /**< Agent population size counter */ 
int h_xmachine_memory_flood_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_flood_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_flood_values;  /**< Agent sort identifiers value */

/* flood state variables */
xmachine_memory_flood_list* h_floods_static;      /**< Pointer to agent list (population) on host*/
xmachine_memory_flood_list* d_floods_static;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_flood_static_count;   /**< Agent population size counter */ 

/* warning Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_warning_list* d_warnings;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_warning_list* d_warnings_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_warning_list* d_warnings_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_warning_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_warning_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_warning_values;  /**< Agent sort identifiers value */

/* warning state variables */
xmachine_memory_warning_list* h_warnings_static_warning;      /**< Pointer to agent list (population) on host*/
xmachine_memory_warning_list* d_warnings_static_warning;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_warning_static_warning_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_households_default_variable_x_data_iteration;
unsigned int h_households_default_variable_y_data_iteration;
unsigned int h_households_default_variable_resident_num_data_iteration;
unsigned int h_households_default_variable_OYI_data_iteration;
unsigned int h_households_default_variable_tenure_data_iteration;
unsigned int h_households_default_variable_take_measure_data_iteration;
unsigned int h_households_default_variable_warning_area_data_iteration;
unsigned int h_households_default_variable_get_warning_data_iteration;
unsigned int h_households_default_variable_alert_state_data_iteration;
unsigned int h_households_default_variable_sandbag_state_data_iteration;
unsigned int h_households_default_variable_sandbag_time_count_data_iteration;
unsigned int h_households_default_variable_flooded_time_data_iteration;
unsigned int h_households_default_variable_initial_wl_data_iteration;
unsigned int h_households_default_variable_actual_wl_data_iteration;
unsigned int h_households_default_variable_average_wl_data_iteration;
unsigned int h_households_default_variable_max_wl_data_iteration;
unsigned int h_households_default_variable_financial_damage_data_iteration;
unsigned int h_households_default_variable_inform_others_data_iteration;
unsigned int h_households_default_variable_get_informed_data_iteration;
unsigned int h_households_default_variable_lod_data_iteration;
unsigned int h_households_default_variable_animate_data_iteration;
unsigned int h_households_default_variable_animate_dir_data_iteration;
unsigned int h_floods_static_variable_x_data_iteration;
unsigned int h_floods_static_variable_y_data_iteration;
unsigned int h_floods_static_variable_floodID_data_iteration;
unsigned int h_floods_static_variable_flood_h_data_iteration;
unsigned int h_warnings_static_warning_variable_x_data_iteration;
unsigned int h_warnings_static_warning_variable_y_data_iteration;
unsigned int h_warnings_static_warning_variable_flooded_households_data_iteration;
unsigned int h_warnings_static_warning_variable_total_financial_damage_data_iteration;
unsigned int h_warnings_static_warning_variable_total_take_measure_data_iteration;
unsigned int h_warnings_static_warning_variable_total_get_warning_data_iteration;
unsigned int h_warnings_static_warning_variable_total_alert_state_data_iteration;
unsigned int h_warnings_static_warning_variable_total_sandbag1_data_iteration;
unsigned int h_warnings_static_warning_variable_total_sandbag2_data_iteration;
unsigned int h_warnings_static_warning_variable_total_sandbag3_data_iteration;
unsigned int h_warnings_static_warning_variable_total_inform_others_data_iteration;
unsigned int h_warnings_static_warning_variable_total_get_informed_data_iteration;


/* Message Memory */

/* flood_cell Message variables */
xmachine_message_flood_cell_list* h_flood_cells;         /**< Pointer to message list on host*/
xmachine_message_flood_cell_list* d_flood_cells;         /**< Pointer to message list on device*/
xmachine_message_flood_cell_list* d_flood_cells_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_flood_cell_range;     /**< range of the discrete message*/
int h_message_flood_cell_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_flood_cell_x_offset;
int h_tex_xmachine_message_flood_cell_y_offset;
int h_tex_xmachine_message_flood_cell_floodID_offset;
int h_tex_xmachine_message_flood_cell_flood_h_offset;
/* financial_damage_infor Message variables */
xmachine_message_financial_damage_infor_list* h_financial_damage_infors;         /**< Pointer to message list on host*/
xmachine_message_financial_damage_infor_list* d_financial_damage_infors;         /**< Pointer to message list on device*/
xmachine_message_financial_damage_infor_list* d_financial_damage_infors_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_financial_damage_infor_count;         /**< message list counter*/
int h_message_financial_damage_infor_output_type;   /**< message output type (single or optional)*/

/* state_data Message variables */
xmachine_message_state_data_list* h_state_datas;         /**< Pointer to message list on host*/
xmachine_message_state_data_list* d_state_datas;         /**< Pointer to message list on device*/
xmachine_message_state_data_list* d_state_datas_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_state_data_count;         /**< message list counter*/
int h_message_state_data_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_household;
size_t temp_scan_storage_bytes_household;

void * d_temp_scan_storage_flood;
size_t temp_scan_storage_bytes_flood;

void * d_temp_scan_storage_warning;
size_t temp_scan_storage_bytes_warning;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** household_output_financial_damage_infor
 * Agent function prototype for output_financial_damage_infor function of household agent
 */
void household_output_financial_damage_infor(cudaStream_t &stream);

/** household_identify_flood
 * Agent function prototype for identify_flood function of household agent
 */
void household_identify_flood(cudaStream_t &stream);

/** household_detect_flood
 * Agent function prototype for detect_flood function of household agent
 */
void household_detect_flood(cudaStream_t &stream);

/** household_communicate
 * Agent function prototype for communicate function of household agent
 */
void household_communicate(cudaStream_t &stream);

/** flood_output_flood_cells
 * Agent function prototype for output_flood_cells function of flood agent
 */
void flood_output_flood_cells(cudaStream_t &stream);

/** flood_generate_warnings
 * Agent function prototype for generate_warnings function of flood agent
 */
void flood_generate_warnings(cudaStream_t &stream);

/** flood_update_data
 * Agent function prototype for update_data function of flood agent
 */
void flood_update_data(cudaStream_t &stream);

/** warning_calcu_damage_infor
 * Agent function prototype for calcu_damage_infor function of warning agent
 */
void warning_calcu_damage_infor(cudaStream_t &stream);

/** warning_output_state_data
 * Agent function prototype for output_state_data function of warning agent
 */
void warning_output_state_data(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_households_default_variable_x_data_iteration = 0;
    h_households_default_variable_y_data_iteration = 0;
    h_households_default_variable_resident_num_data_iteration = 0;
    h_households_default_variable_OYI_data_iteration = 0;
    h_households_default_variable_tenure_data_iteration = 0;
    h_households_default_variable_take_measure_data_iteration = 0;
    h_households_default_variable_warning_area_data_iteration = 0;
    h_households_default_variable_get_warning_data_iteration = 0;
    h_households_default_variable_alert_state_data_iteration = 0;
    h_households_default_variable_sandbag_state_data_iteration = 0;
    h_households_default_variable_sandbag_time_count_data_iteration = 0;
    h_households_default_variable_flooded_time_data_iteration = 0;
    h_households_default_variable_initial_wl_data_iteration = 0;
    h_households_default_variable_actual_wl_data_iteration = 0;
    h_households_default_variable_average_wl_data_iteration = 0;
    h_households_default_variable_max_wl_data_iteration = 0;
    h_households_default_variable_financial_damage_data_iteration = 0;
    h_households_default_variable_inform_others_data_iteration = 0;
    h_households_default_variable_get_informed_data_iteration = 0;
    h_households_default_variable_lod_data_iteration = 0;
    h_households_default_variable_animate_data_iteration = 0;
    h_households_default_variable_animate_dir_data_iteration = 0;
    h_floods_static_variable_x_data_iteration = 0;
    h_floods_static_variable_y_data_iteration = 0;
    h_floods_static_variable_floodID_data_iteration = 0;
    h_floods_static_variable_flood_h_data_iteration = 0;
    h_warnings_static_warning_variable_x_data_iteration = 0;
    h_warnings_static_warning_variable_y_data_iteration = 0;
    h_warnings_static_warning_variable_flooded_households_data_iteration = 0;
    h_warnings_static_warning_variable_total_financial_damage_data_iteration = 0;
    h_warnings_static_warning_variable_total_take_measure_data_iteration = 0;
    h_warnings_static_warning_variable_total_get_warning_data_iteration = 0;
    h_warnings_static_warning_variable_total_alert_state_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag1_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag2_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag3_data_iteration = 0;
    h_warnings_static_warning_variable_total_inform_others_data_iteration = 0;
    h_warnings_static_warning_variable_total_get_informed_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_household_SoA_size = sizeof(xmachine_memory_household_list);
	h_households_default = (xmachine_memory_household_list*)malloc(xmachine_household_SoA_size);
	int xmachine_flood_SoA_size = sizeof(xmachine_memory_flood_list);
	h_floods_static = (xmachine_memory_flood_list*)malloc(xmachine_flood_SoA_size);
	int xmachine_warning_SoA_size = sizeof(xmachine_memory_warning_list);
	h_warnings_static_warning = (xmachine_memory_warning_list*)malloc(xmachine_warning_SoA_size);

	/* Message memory allocation (CPU) */
	int message_flood_cell_SoA_size = sizeof(xmachine_message_flood_cell_list);
	h_flood_cells = (xmachine_message_flood_cell_list*)malloc(message_flood_cell_SoA_size);
	int message_financial_damage_infor_SoA_size = sizeof(xmachine_message_financial_damage_infor_list);
	h_financial_damage_infors = (xmachine_message_financial_damage_infor_list*)malloc(message_financial_damage_infor_SoA_size);
	int message_state_data_SoA_size = sizeof(xmachine_message_state_data_list);
	h_state_datas = (xmachine_message_state_data_list*)malloc(message_state_data_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

	/* Graph memory allocation (CPU) */
	

    PROFILE_POP_RANGE(); //"allocate host"
	
	
	/* Set discrete flood_cell message variables (range, width)*/
	h_message_flood_cell_range = 0; //from xml
	h_message_flood_cell_width = (int)floor(sqrt((float)xmachine_message_flood_cell_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_flood_cell_MAX)){
		printf("ERROR: flood_cell message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_flood_cell_range, &h_message_flood_cell_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_flood_cell_width, &h_message_flood_cell_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_flood_MAX)){
		printf("ERROR: floods agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_flood_pop_width = (int)sqrt(xmachine_memory_flood_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_households_default, &h_xmachine_memory_household_default_count, h_floods_static, &h_xmachine_memory_flood_static_count, h_warnings_static_warning, &h_xmachine_memory_warning_static_warning_count);

	// Read graphs from disk
	

  PROFILE_PUSH_RANGE("allocate device");
  
	
	/* household Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_households, xmachine_household_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_households_swap, xmachine_household_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_households_new, xmachine_household_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_household_keys, xmachine_memory_household_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_household_values, xmachine_memory_household_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_households_default, xmachine_household_SoA_size));
	gpuErrchk( cudaMemcpy( d_households_default, h_households_default, xmachine_household_SoA_size, cudaMemcpyHostToDevice));
    
	/* flood Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_floods, xmachine_flood_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_floods_swap, xmachine_flood_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_floods_new, xmachine_flood_SoA_size));
    
	/* static memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_floods_static, xmachine_flood_SoA_size));
	gpuErrchk( cudaMemcpy( d_floods_static, h_floods_static, xmachine_flood_SoA_size, cudaMemcpyHostToDevice));
    
	/* warning Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_warnings, xmachine_warning_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_warnings_swap, xmachine_warning_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_warnings_new, xmachine_warning_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_warning_keys, xmachine_memory_warning_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_warning_values, xmachine_memory_warning_MAX* sizeof(uint)));
	/* static_warning memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_warnings_static_warning, xmachine_warning_SoA_size));
	gpuErrchk( cudaMemcpy( d_warnings_static_warning, h_warnings_static_warning, xmachine_warning_SoA_size, cudaMemcpyHostToDevice));
    
	/* flood_cell Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_flood_cells, message_flood_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_flood_cells_swap, message_flood_cell_SoA_size));
	gpuErrchk( cudaMemcpy( d_flood_cells, h_flood_cells, message_flood_cell_SoA_size, cudaMemcpyHostToDevice));
	
	/* financial_damage_infor Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_financial_damage_infors, message_financial_damage_infor_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_financial_damage_infors_swap, message_financial_damage_infor_SoA_size));
	gpuErrchk( cudaMemcpy( d_financial_damage_infors, h_financial_damage_infors, message_financial_damage_infor_SoA_size, cudaMemcpyHostToDevice));
	
	/* state_data Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_state_datas, message_state_data_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_state_datas_swap, message_state_data_SoA_size));
	gpuErrchk( cudaMemcpy( d_state_datas, h_state_datas, message_state_data_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_household = nullptr;
    temp_scan_storage_bytes_household = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_household, 
        temp_scan_storage_bytes_household, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_household_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_household, temp_scan_storage_bytes_household));
    
    d_temp_scan_storage_flood = nullptr;
    temp_scan_storage_bytes_flood = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_flood, 
        temp_scan_storage_bytes_flood, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_flood_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_flood, temp_scan_storage_bytes_flood));
    
    d_temp_scan_storage_warning = nullptr;
    temp_scan_storage_bytes_warning = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_warning, 
        temp_scan_storage_bytes_warning, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_warning_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_warning, temp_scan_storage_bytes_warning));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();
	
	
	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    initConstants();
    PROFILE_PUSH_RANGE("initConstants");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initConstants = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
  gpuErrchk(cudaStreamCreate(&stream2));
  gpuErrchk(cudaStreamCreate(&stream3));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_household_default_count: %u\n",get_agent_household_default_count());
	
		printf("Init agent_flood_static_count: %u\n",get_agent_flood_static_count());
	
		printf("Init agent_warning_static_warning_count: %u\n",get_agent_warning_static_warning_count());
	
#endif
} 


void sort_households_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_household_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_household_default_count); 
	gridSize = (h_xmachine_memory_household_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_household_keys, d_xmachine_memory_household_values, d_households_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_household_keys),  thrust::device_pointer_cast(d_xmachine_memory_household_keys) + h_xmachine_memory_household_default_count,  thrust::device_pointer_cast(d_xmachine_memory_household_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_household_agents, no_sm, h_xmachine_memory_household_default_count); 
	gridSize = (h_xmachine_memory_household_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_household_agents<<<gridSize, blockSize>>>(d_xmachine_memory_household_values, d_households_default, d_households_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_household_list* d_households_temp = d_households_default;
	d_households_default = d_households_swap;
	d_households_swap = d_households_temp;	
}

void sort_warnings_static_warning(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_warning_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_warning_static_warning_count); 
	gridSize = (h_xmachine_memory_warning_static_warning_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_warning_keys, d_xmachine_memory_warning_values, d_warnings_static_warning);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_warning_keys),  thrust::device_pointer_cast(d_xmachine_memory_warning_keys) + h_xmachine_memory_warning_static_warning_count,  thrust::device_pointer_cast(d_xmachine_memory_warning_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_warning_agents, no_sm, h_xmachine_memory_warning_static_warning_count); 
	gridSize = (h_xmachine_memory_warning_static_warning_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_warning_agents<<<gridSize, blockSize>>>(d_xmachine_memory_warning_values, d_warnings_static_warning, d_warnings_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_warning_list* d_warnings_temp = d_warnings_static_warning;
	d_warnings_static_warning = d_warnings_swap;
	d_warnings_swap = d_warnings_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* household Agent variables */
	gpuErrchk(cudaFree(d_households));
	gpuErrchk(cudaFree(d_households_swap));
	gpuErrchk(cudaFree(d_households_new));
	
	free( h_households_default);
	gpuErrchk(cudaFree(d_households_default));
	
	/* flood Agent variables */
	gpuErrchk(cudaFree(d_floods));
	gpuErrchk(cudaFree(d_floods_swap));
	gpuErrchk(cudaFree(d_floods_new));
	
	free( h_floods_static);
	gpuErrchk(cudaFree(d_floods_static));
	
	/* warning Agent variables */
	gpuErrchk(cudaFree(d_warnings));
	gpuErrchk(cudaFree(d_warnings_swap));
	gpuErrchk(cudaFree(d_warnings_new));
	
	free( h_warnings_static_warning);
	gpuErrchk(cudaFree(d_warnings_static_warning));
	

	/* Message data free */
	
	/* flood_cell Message variables */
	free( h_flood_cells);
	gpuErrchk(cudaFree(d_flood_cells));
	gpuErrchk(cudaFree(d_flood_cells_swap));
	
	/* financial_damage_infor Message variables */
	free( h_financial_damage_infors);
	gpuErrchk(cudaFree(d_financial_damage_infors));
	gpuErrchk(cudaFree(d_financial_damage_infors_swap));
	
	/* state_data Message variables */
	free( h_state_datas);
	gpuErrchk(cudaFree(d_state_datas));
	gpuErrchk(cudaFree(d_state_datas_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_household != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_household));
      d_temp_scan_storage_household = nullptr;
      temp_scan_storage_bytes_household = 0;
    }
    
    if(d_temp_scan_storage_flood != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_flood));
      d_temp_scan_storage_flood = nullptr;
      temp_scan_storage_bytes_flood = 0;
    }
    
    if(d_temp_scan_storage_warning != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_warning));
      d_temp_scan_storage_warning = nullptr;
      temp_scan_storage_bytes_warning = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/
	h_message_financial_damage_infor_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_financial_damage_infor_count, &h_message_financial_damage_infor_count, sizeof(int)));
	
	h_message_state_data_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_state_data_count, &h_message_state_data_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("flood_generate_warnings");
	flood_generate_warnings(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: flood_generate_warnings = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("flood_update_data");
	flood_update_data(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: flood_update_data = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("flood_output_flood_cells");
	flood_output_flood_cells(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: flood_output_flood_cells = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("household_output_financial_damage_infor");
	household_output_financial_damage_infor(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: household_output_financial_damage_infor = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("warning_output_state_data");
	warning_output_state_data(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: warning_output_state_data = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("warning_calcu_damage_infor");
	warning_calcu_damage_infor(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: warning_calcu_damage_infor = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("household_identify_flood");
	household_identify_flood(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: household_identify_flood = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("household_detect_flood");
	household_detect_flood(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: household_detect_flood = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("household_communicate");
	household_communicate(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: household_communicate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    PROFILE_PUSH_RANGE("read_data_func");
	read_data_func();
	
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: read_data_func = %f (ms)\n", instrument_milliseconds);
#endif

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_household_default_count: %u\n",get_agent_household_default_count());
	
		printf("agent_flood_static_count: %u\n",get_agent_flood_static_count());
	
		printf("agent_warning_static_warning_count: %u\n",get_agent_warning_static_warning_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration
short h_env_FLOOD_DATA_ARRAY[900000];
int h_env_TIME;
int h_env_RANDOM_SEED_SEC;
int h_env_RANDOM_SEED_MIN;
float h_env_TIME_SCALER;

void init_flood_data_array(){
	if(FLOOD_DATA_ARRAY == nullptr){
		cudaMalloc((void **)&FLOOD_DATA_ARRAY, 900000 * sizeof(short));
		printf("address of flood data arrat %p",FLOOD_DATA_ARRAY);
	}
	}



//constant setter
void set_FLOOD_DATA_ARRAY(short* h_FLOOD_DATA_ARRAY){

	
	init_flood_data_array();
	
	gpuErrchk(cudaMemcpy(FLOOD_DATA_ARRAY, h_FLOOD_DATA_ARRAY, 900000 * sizeof(short), cudaMemcpyHostToDevice));

	

	memcpy(&h_env_FLOOD_DATA_ARRAY, h_FLOOD_DATA_ARRAY,sizeof(short)*900000);
}

//constant getter
const short* get_FLOOD_DATA_ARRAY(){
    return h_env_FLOOD_DATA_ARRAY;
}




//constant setter
void set_TIME(int* h_TIME){

	gpuErrchk(cudaMemcpyToSymbol(TIME, h_TIME, sizeof(int)));

	

	memcpy(&h_env_TIME, h_TIME,sizeof(int));
}

//constant getter
const int* get_TIME(){
    return &h_env_TIME;
}




//constant setter
void set_RANDOM_SEED_SEC(int* h_RANDOM_SEED_SEC){

	gpuErrchk(cudaMemcpyToSymbol(RANDOM_SEED_SEC, h_RANDOM_SEED_SEC, sizeof(int)));

	

	memcpy(&h_env_RANDOM_SEED_SEC, h_RANDOM_SEED_SEC,sizeof(int));
}

//constant getter
const int* get_RANDOM_SEED_SEC(){
    return &h_env_RANDOM_SEED_SEC;
}




//constant setter
void set_RANDOM_SEED_MIN(int* h_RANDOM_SEED_MIN){

	gpuErrchk(cudaMemcpyToSymbol(RANDOM_SEED_MIN, h_RANDOM_SEED_MIN, sizeof(int)));

	

	memcpy(&h_env_RANDOM_SEED_MIN, h_RANDOM_SEED_MIN,sizeof(int));
}

//constant getter
const int* get_RANDOM_SEED_MIN(){
    return &h_env_RANDOM_SEED_MIN;
}




//constant setter
void set_TIME_SCALER(float* h_TIME_SCALER){

	gpuErrchk(cudaMemcpyToSymbol(TIME_SCALER, h_TIME_SCALER, sizeof(float)));

	

	memcpy(&h_env_TIME_SCALER, h_TIME_SCALER,sizeof(float));
}

//constant getter
const float* get_TIME_SCALER(){
    return &h_env_TIME_SCALER;
}




/* Agent data access functions*/

    
int get_agent_household_MAX_count(){
    return xmachine_memory_household_MAX;
}


int get_agent_household_default_count(){
	//continuous agent
	return h_xmachine_memory_household_default_count;
	
}

xmachine_memory_household_list* get_device_household_default_agents(){
	return d_households_default;
}

xmachine_memory_household_list* get_host_household_default_agents(){
	return h_households_default;
}

    
int get_agent_flood_MAX_count(){
    return xmachine_memory_flood_MAX;
}


int get_agent_flood_static_count(){
	//discrete agent 
	return xmachine_memory_flood_MAX;
}

xmachine_memory_flood_list* get_device_flood_static_agents(){
	return d_floods_static;
}

xmachine_memory_flood_list* get_host_flood_static_agents(){
	return h_floods_static;
}

int get_flood_population_width(){
  return h_xmachine_memory_flood_pop_width;
}

    
int get_agent_warning_MAX_count(){
    return xmachine_memory_warning_MAX;
}


int get_agent_warning_static_warning_count(){
	//continuous agent
	return h_xmachine_memory_warning_static_warning_count;
	
}

xmachine_memory_warning_list* get_device_warning_static_warning_agents(){
	return d_warnings_static_warning;
}

xmachine_memory_warning_list* get_host_warning_static_warning_agents(){
	return h_warnings_static_warning;
}



/* Host based access of agent variables*/

/** float get_household_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_household_default_variable_x(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->x,
                    d_households_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_household_default_variable_y(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->y,
                    d_households_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_resident_num(unsigned int index)
 * Gets the value of the resident_num variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable resident_num
 */
__host__ int get_household_default_variable_resident_num(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_resident_num_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->resident_num,
                    d_households_default->resident_num,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_resident_num_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->resident_num[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access resident_num for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_OYI(unsigned int index)
 * Gets the value of the OYI variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable OYI
 */
__host__ int get_household_default_variable_OYI(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_OYI_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->OYI,
                    d_households_default->OYI,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_OYI_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->OYI[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access OYI for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_tenure(unsigned int index)
 * Gets the value of the tenure variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tenure
 */
__host__ int get_household_default_variable_tenure(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_tenure_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->tenure,
                    d_households_default->tenure,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_tenure_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->tenure[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tenure for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_take_measure(unsigned int index)
 * Gets the value of the take_measure variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable take_measure
 */
__host__ int get_household_default_variable_take_measure(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_take_measure_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->take_measure,
                    d_households_default->take_measure,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_take_measure_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->take_measure[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access take_measure for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_warning_area(unsigned int index)
 * Gets the value of the warning_area variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable warning_area
 */
__host__ int get_household_default_variable_warning_area(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_warning_area_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->warning_area,
                    d_households_default->warning_area,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_warning_area_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->warning_area[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access warning_area for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_get_warning(unsigned int index)
 * Gets the value of the get_warning variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable get_warning
 */
__host__ int get_household_default_variable_get_warning(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_get_warning_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->get_warning,
                    d_households_default->get_warning,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_get_warning_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->get_warning[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access get_warning for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_alert_state(unsigned int index)
 * Gets the value of the alert_state variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable alert_state
 */
__host__ int get_household_default_variable_alert_state(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_alert_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->alert_state,
                    d_households_default->alert_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_alert_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->alert_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access alert_state for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_sandbag_state(unsigned int index)
 * Gets the value of the sandbag_state variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sandbag_state
 */
__host__ int get_household_default_variable_sandbag_state(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_sandbag_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->sandbag_state,
                    d_households_default->sandbag_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_sandbag_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->sandbag_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access sandbag_state for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_sandbag_time_count(unsigned int index)
 * Gets the value of the sandbag_time_count variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sandbag_time_count
 */
__host__ int get_household_default_variable_sandbag_time_count(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_sandbag_time_count_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->sandbag_time_count,
                    d_households_default->sandbag_time_count,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_sandbag_time_count_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->sandbag_time_count[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access sandbag_time_count for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_flooded_time(unsigned int index)
 * Gets the value of the flooded_time variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flooded_time
 */
__host__ int get_household_default_variable_flooded_time(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_flooded_time_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->flooded_time,
                    d_households_default->flooded_time,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_flooded_time_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->flooded_time[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access flooded_time for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_initial_wl(unsigned int index)
 * Gets the value of the initial_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable initial_wl
 */
__host__ float get_household_default_variable_initial_wl(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_initial_wl_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->initial_wl,
                    d_households_default->initial_wl,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_initial_wl_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->initial_wl[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access initial_wl for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_actual_wl(unsigned int index)
 * Gets the value of the actual_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable actual_wl
 */
__host__ float get_household_default_variable_actual_wl(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_actual_wl_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->actual_wl,
                    d_households_default->actual_wl,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_actual_wl_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->actual_wl[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access actual_wl for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_average_wl(unsigned int index)
 * Gets the value of the average_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable average_wl
 */
__host__ float get_household_default_variable_average_wl(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_average_wl_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->average_wl,
                    d_households_default->average_wl,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_average_wl_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->average_wl[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access average_wl for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_max_wl(unsigned int index)
 * Gets the value of the max_wl variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable max_wl
 */
__host__ float get_household_default_variable_max_wl(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_max_wl_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->max_wl,
                    d_households_default->max_wl,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_max_wl_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->max_wl[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access max_wl for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_financial_damage(unsigned int index)
 * Gets the value of the financial_damage variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable financial_damage
 */
__host__ float get_household_default_variable_financial_damage(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_financial_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->financial_damage,
                    d_households_default->financial_damage,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_financial_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->financial_damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access financial_damage for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_inform_others(unsigned int index)
 * Gets the value of the inform_others variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable inform_others
 */
__host__ int get_household_default_variable_inform_others(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_inform_others_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->inform_others,
                    d_households_default->inform_others,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_inform_others_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->inform_others[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access inform_others for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_get_informed(unsigned int index)
 * Gets the value of the get_informed variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable get_informed
 */
__host__ int get_household_default_variable_get_informed(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_get_informed_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->get_informed,
                    d_households_default->get_informed,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_get_informed_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->get_informed[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access get_informed for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_household_default_variable_lod(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_lod_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->lod,
                    d_households_default->lod,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_lod_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->lod[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lod for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_household_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_household_default_variable_animate(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_animate_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->animate,
                    d_households_default->animate,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_animate_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->animate[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_household_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an household agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_household_default_variable_animate_dir(unsigned int index){
    unsigned int count = get_agent_household_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_households_default_variable_animate_dir_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_households_default->animate_dir,
                    d_households_default->animate_dir,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_households_default_variable_animate_dir_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_households_default->animate_dir[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate_dir for the %u th member of household_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_flood_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_flood_static_variable_x(unsigned int index){
    unsigned int count = get_agent_flood_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_floods_static_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_floods_static->x,
                    d_floods_static->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_floods_static_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_floods_static->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of flood_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_flood_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_flood_static_variable_y(unsigned int index){
    unsigned int count = get_agent_flood_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_floods_static_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_floods_static->y,
                    d_floods_static->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_floods_static_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_floods_static->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of flood_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_flood_static_variable_floodID(unsigned int index)
 * Gets the value of the floodID variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable floodID
 */
__host__ int get_flood_static_variable_floodID(unsigned int index){
    unsigned int count = get_agent_flood_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_floods_static_variable_floodID_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_floods_static->floodID,
                    d_floods_static->floodID,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_floods_static_variable_floodID_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_floods_static->floodID[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access floodID for the %u th member of flood_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_flood_static_variable_flood_h(unsigned int index)
 * Gets the value of the flood_h variable of an flood agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flood_h
 */
__host__ float get_flood_static_variable_flood_h(unsigned int index){
    unsigned int count = get_agent_flood_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_floods_static_variable_flood_h_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_floods_static->flood_h,
                    d_floods_static->flood_h,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_floods_static_variable_flood_h_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_floods_static->flood_h[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access flood_h for the %u th member of flood_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_warning_static_warning_variable_x(unsigned int index)
 * Gets the value of the x variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_warning_static_warning_variable_x(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->x,
                    d_warnings_static_warning->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_warning_static_warning_variable_y(unsigned int index)
 * Gets the value of the y variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_warning_static_warning_variable_y(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->y,
                    d_warnings_static_warning->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_flooded_households(unsigned int index)
 * Gets the value of the flooded_households variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable flooded_households
 */
__host__ int get_warning_static_warning_variable_flooded_households(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_flooded_households_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->flooded_households,
                    d_warnings_static_warning->flooded_households,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_flooded_households_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->flooded_households[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access flooded_households for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_warning_static_warning_variable_total_financial_damage(unsigned int index)
 * Gets the value of the total_financial_damage variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_financial_damage
 */
__host__ float get_warning_static_warning_variable_total_financial_damage(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_financial_damage_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_financial_damage,
                    d_warnings_static_warning->total_financial_damage,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_financial_damage_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_financial_damage[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_financial_damage for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_take_measure(unsigned int index)
 * Gets the value of the total_take_measure variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_take_measure
 */
__host__ int get_warning_static_warning_variable_total_take_measure(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_take_measure_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_take_measure,
                    d_warnings_static_warning->total_take_measure,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_take_measure_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_take_measure[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_take_measure for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_get_warning(unsigned int index)
 * Gets the value of the total_get_warning variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_get_warning
 */
__host__ int get_warning_static_warning_variable_total_get_warning(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_get_warning_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_get_warning,
                    d_warnings_static_warning->total_get_warning,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_get_warning_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_get_warning[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_get_warning for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_alert_state(unsigned int index)
 * Gets the value of the total_alert_state variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_alert_state
 */
__host__ int get_warning_static_warning_variable_total_alert_state(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_alert_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_alert_state,
                    d_warnings_static_warning->total_alert_state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_alert_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_alert_state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_alert_state for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_sandbag1(unsigned int index)
 * Gets the value of the total_sandbag1 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag1
 */
__host__ int get_warning_static_warning_variable_total_sandbag1(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_sandbag1_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_sandbag1,
                    d_warnings_static_warning->total_sandbag1,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_sandbag1_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_sandbag1[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_sandbag1 for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_sandbag2(unsigned int index)
 * Gets the value of the total_sandbag2 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag2
 */
__host__ int get_warning_static_warning_variable_total_sandbag2(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_sandbag2_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_sandbag2,
                    d_warnings_static_warning->total_sandbag2,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_sandbag2_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_sandbag2[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_sandbag2 for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_sandbag3(unsigned int index)
 * Gets the value of the total_sandbag3 variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_sandbag3
 */
__host__ int get_warning_static_warning_variable_total_sandbag3(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_sandbag3_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_sandbag3,
                    d_warnings_static_warning->total_sandbag3,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_sandbag3_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_sandbag3[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_sandbag3 for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_inform_others(unsigned int index)
 * Gets the value of the total_inform_others variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_inform_others
 */
__host__ int get_warning_static_warning_variable_total_inform_others(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_inform_others_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_inform_others,
                    d_warnings_static_warning->total_inform_others,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_inform_others_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_inform_others[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_inform_others for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_warning_static_warning_variable_total_get_informed(unsigned int index)
 * Gets the value of the total_get_informed variable of an warning agent in the static_warning state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable total_get_informed
 */
__host__ int get_warning_static_warning_variable_total_get_informed(unsigned int index){
    unsigned int count = get_agent_warning_static_warning_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_warnings_static_warning_variable_total_get_informed_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_warnings_static_warning->total_get_informed,
                    d_warnings_static_warning->total_get_informed,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_warnings_static_warning_variable_total_get_informed_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_warnings_static_warning->total_get_informed[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access total_get_informed for the %u th member of warning_static_warning. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_household_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_household_hostToDevice(xmachine_memory_household_list * d_dst, xmachine_memory_household * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->resident_num, &h_agent->resident_num, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->OYI, &h_agent->OYI, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tenure, &h_agent->tenure, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->take_measure, &h_agent->take_measure, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->warning_area, &h_agent->warning_area, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->get_warning, &h_agent->get_warning, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->alert_state, &h_agent->alert_state, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->sandbag_state, &h_agent->sandbag_state, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->sandbag_time_count, &h_agent->sandbag_time_count, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->flooded_time, &h_agent->flooded_time, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->initial_wl, &h_agent->initial_wl, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->actual_wl, &h_agent->actual_wl, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->average_wl, &h_agent->average_wl, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->max_wl, &h_agent->max_wl, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->financial_damage, &h_agent->financial_damage, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->inform_others, &h_agent->inform_others, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->get_informed, &h_agent->get_informed, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, &h_agent->lod, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, &h_agent->animate, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, &h_agent->animate_dir, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_household_hostToDevice(xmachine_memory_household_list * d_dst, xmachine_memory_household_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->resident_num, h_src->resident_num, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->OYI, h_src->OYI, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tenure, h_src->tenure, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->take_measure, h_src->take_measure, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->warning_area, h_src->warning_area, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->get_warning, h_src->get_warning, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->alert_state, h_src->alert_state, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->sandbag_state, h_src->sandbag_state, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->sandbag_time_count, h_src->sandbag_time_count, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->flooded_time, h_src->flooded_time, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->initial_wl, h_src->initial_wl, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->actual_wl, h_src->actual_wl, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->average_wl, h_src->average_wl, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->max_wl, h_src->max_wl, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->financial_damage, h_src->financial_damage, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->inform_others, h_src->inform_others, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->get_informed, h_src->get_informed, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, h_src->lod, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, h_src->animate, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, h_src->animate_dir, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_warning_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_warning_hostToDevice(xmachine_memory_warning_list * d_dst, xmachine_memory_warning * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->flooded_households, &h_agent->flooded_households, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_financial_damage, &h_agent->total_financial_damage, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_take_measure, &h_agent->total_take_measure, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_get_warning, &h_agent->total_get_warning, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_alert_state, &h_agent->total_alert_state, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag1, &h_agent->total_sandbag1, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag2, &h_agent->total_sandbag2, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag3, &h_agent->total_sandbag3, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_inform_others, &h_agent->total_inform_others, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_get_informed, &h_agent->total_get_informed, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_warning_hostToDevice(xmachine_memory_warning_list * d_dst, xmachine_memory_warning_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->flooded_households, h_src->flooded_households, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_financial_damage, h_src->total_financial_damage, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_take_measure, h_src->total_take_measure, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_get_warning, h_src->total_get_warning, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_alert_state, h_src->total_alert_state, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag1, h_src->total_sandbag1, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag2, h_src->total_sandbag2, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_sandbag3, h_src->total_sandbag3, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_inform_others, h_src->total_inform_others, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->total_get_informed, h_src->total_get_informed, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_household* h_allocate_agent_household(){
	xmachine_memory_household* agent = (xmachine_memory_household*)malloc(sizeof(xmachine_memory_household));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_household));

	return agent;
}
void h_free_agent_household(xmachine_memory_household** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_household** h_allocate_agent_household_array(unsigned int count){
	xmachine_memory_household ** agents = (xmachine_memory_household**)malloc(count * sizeof(xmachine_memory_household*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_household();
	}
	return agents;
}
void h_free_agent_household_array(xmachine_memory_household*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_household(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_household_AoS_to_SoA(xmachine_memory_household_list * dst, xmachine_memory_household** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->resident_num[i] = src[i]->resident_num;
			 
			dst->OYI[i] = src[i]->OYI;
			 
			dst->tenure[i] = src[i]->tenure;
			 
			dst->take_measure[i] = src[i]->take_measure;
			 
			dst->warning_area[i] = src[i]->warning_area;
			 
			dst->get_warning[i] = src[i]->get_warning;
			 
			dst->alert_state[i] = src[i]->alert_state;
			 
			dst->sandbag_state[i] = src[i]->sandbag_state;
			 
			dst->sandbag_time_count[i] = src[i]->sandbag_time_count;
			 
			dst->flooded_time[i] = src[i]->flooded_time;
			 
			dst->initial_wl[i] = src[i]->initial_wl;
			 
			dst->actual_wl[i] = src[i]->actual_wl;
			 
			dst->average_wl[i] = src[i]->average_wl;
			 
			dst->max_wl[i] = src[i]->max_wl;
			 
			dst->financial_damage[i] = src[i]->financial_damage;
			 
			dst->inform_others[i] = src[i]->inform_others;
			 
			dst->get_informed[i] = src[i]->get_informed;
			 
			dst->lod[i] = src[i]->lod;
			 
			dst->animate[i] = src[i]->animate;
			 
			dst->animate_dir[i] = src[i]->animate_dir;
			
		}
	}
}


void h_add_agent_household_default(xmachine_memory_household* agent){
	if (h_xmachine_memory_household_count + 1 > xmachine_memory_household_MAX){
		printf("Error: Buffer size of household agents in state default will be exceeded by h_add_agent_household_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_household_hostToDevice(d_households_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_household_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_household_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_households_default, d_households_new, h_xmachine_memory_household_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_household_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_households_default_variable_x_data_iteration = 0;
    h_households_default_variable_y_data_iteration = 0;
    h_households_default_variable_resident_num_data_iteration = 0;
    h_households_default_variable_OYI_data_iteration = 0;
    h_households_default_variable_tenure_data_iteration = 0;
    h_households_default_variable_take_measure_data_iteration = 0;
    h_households_default_variable_warning_area_data_iteration = 0;
    h_households_default_variable_get_warning_data_iteration = 0;
    h_households_default_variable_alert_state_data_iteration = 0;
    h_households_default_variable_sandbag_state_data_iteration = 0;
    h_households_default_variable_sandbag_time_count_data_iteration = 0;
    h_households_default_variable_flooded_time_data_iteration = 0;
    h_households_default_variable_initial_wl_data_iteration = 0;
    h_households_default_variable_actual_wl_data_iteration = 0;
    h_households_default_variable_average_wl_data_iteration = 0;
    h_households_default_variable_max_wl_data_iteration = 0;
    h_households_default_variable_financial_damage_data_iteration = 0;
    h_households_default_variable_inform_others_data_iteration = 0;
    h_households_default_variable_get_informed_data_iteration = 0;
    h_households_default_variable_lod_data_iteration = 0;
    h_households_default_variable_animate_data_iteration = 0;
    h_households_default_variable_animate_dir_data_iteration = 0;
    

}
void h_add_agents_household_default(xmachine_memory_household** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_household_count + count > xmachine_memory_household_MAX){
			printf("Error: Buffer size of household agents in state default will be exceeded by h_add_agents_household_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_household_AoS_to_SoA(h_households_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_household_hostToDevice(d_households_new, h_households_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_household_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_household_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_households_default, d_households_new, h_xmachine_memory_household_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_household_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_households_default_variable_x_data_iteration = 0;
        h_households_default_variable_y_data_iteration = 0;
        h_households_default_variable_resident_num_data_iteration = 0;
        h_households_default_variable_OYI_data_iteration = 0;
        h_households_default_variable_tenure_data_iteration = 0;
        h_households_default_variable_take_measure_data_iteration = 0;
        h_households_default_variable_warning_area_data_iteration = 0;
        h_households_default_variable_get_warning_data_iteration = 0;
        h_households_default_variable_alert_state_data_iteration = 0;
        h_households_default_variable_sandbag_state_data_iteration = 0;
        h_households_default_variable_sandbag_time_count_data_iteration = 0;
        h_households_default_variable_flooded_time_data_iteration = 0;
        h_households_default_variable_initial_wl_data_iteration = 0;
        h_households_default_variable_actual_wl_data_iteration = 0;
        h_households_default_variable_average_wl_data_iteration = 0;
        h_households_default_variable_max_wl_data_iteration = 0;
        h_households_default_variable_financial_damage_data_iteration = 0;
        h_households_default_variable_inform_others_data_iteration = 0;
        h_households_default_variable_get_informed_data_iteration = 0;
        h_households_default_variable_lod_data_iteration = 0;
        h_households_default_variable_animate_data_iteration = 0;
        h_households_default_variable_animate_dir_data_iteration = 0;
        

	}
}

xmachine_memory_warning* h_allocate_agent_warning(){
	xmachine_memory_warning* agent = (xmachine_memory_warning*)malloc(sizeof(xmachine_memory_warning));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_warning));

	return agent;
}
void h_free_agent_warning(xmachine_memory_warning** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_warning** h_allocate_agent_warning_array(unsigned int count){
	xmachine_memory_warning ** agents = (xmachine_memory_warning**)malloc(count * sizeof(xmachine_memory_warning*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_warning();
	}
	return agents;
}
void h_free_agent_warning_array(xmachine_memory_warning*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_warning(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_warning_AoS_to_SoA(xmachine_memory_warning_list * dst, xmachine_memory_warning** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->flooded_households[i] = src[i]->flooded_households;
			 
			dst->total_financial_damage[i] = src[i]->total_financial_damage;
			 
			dst->total_take_measure[i] = src[i]->total_take_measure;
			 
			dst->total_get_warning[i] = src[i]->total_get_warning;
			 
			dst->total_alert_state[i] = src[i]->total_alert_state;
			 
			dst->total_sandbag1[i] = src[i]->total_sandbag1;
			 
			dst->total_sandbag2[i] = src[i]->total_sandbag2;
			 
			dst->total_sandbag3[i] = src[i]->total_sandbag3;
			 
			dst->total_inform_others[i] = src[i]->total_inform_others;
			 
			dst->total_get_informed[i] = src[i]->total_get_informed;
			
		}
	}
}


void h_add_agent_warning_static_warning(xmachine_memory_warning* agent){
	if (h_xmachine_memory_warning_count + 1 > xmachine_memory_warning_MAX){
		printf("Error: Buffer size of warning agents in state static_warning will be exceeded by h_add_agent_warning_static_warning\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_warning_hostToDevice(d_warnings_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_warning_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_warning_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_warnings_static_warning, d_warnings_new, h_xmachine_memory_warning_static_warning_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_warning_static_warning_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_warnings_static_warning_variable_x_data_iteration = 0;
    h_warnings_static_warning_variable_y_data_iteration = 0;
    h_warnings_static_warning_variable_flooded_households_data_iteration = 0;
    h_warnings_static_warning_variable_total_financial_damage_data_iteration = 0;
    h_warnings_static_warning_variable_total_take_measure_data_iteration = 0;
    h_warnings_static_warning_variable_total_get_warning_data_iteration = 0;
    h_warnings_static_warning_variable_total_alert_state_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag1_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag2_data_iteration = 0;
    h_warnings_static_warning_variable_total_sandbag3_data_iteration = 0;
    h_warnings_static_warning_variable_total_inform_others_data_iteration = 0;
    h_warnings_static_warning_variable_total_get_informed_data_iteration = 0;
    

}
void h_add_agents_warning_static_warning(xmachine_memory_warning** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_warning_count + count > xmachine_memory_warning_MAX){
			printf("Error: Buffer size of warning agents in state static_warning will be exceeded by h_add_agents_warning_static_warning\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_warning_AoS_to_SoA(h_warnings_static_warning, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_warning_hostToDevice(d_warnings_new, h_warnings_static_warning, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_warning_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_warning_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_warnings_static_warning, d_warnings_new, h_xmachine_memory_warning_static_warning_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_warning_static_warning_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_warnings_static_warning_variable_x_data_iteration = 0;
        h_warnings_static_warning_variable_y_data_iteration = 0;
        h_warnings_static_warning_variable_flooded_households_data_iteration = 0;
        h_warnings_static_warning_variable_total_financial_damage_data_iteration = 0;
        h_warnings_static_warning_variable_total_take_measure_data_iteration = 0;
        h_warnings_static_warning_variable_total_get_warning_data_iteration = 0;
        h_warnings_static_warning_variable_total_alert_state_data_iteration = 0;
        h_warnings_static_warning_variable_total_sandbag1_data_iteration = 0;
        h_warnings_static_warning_variable_total_sandbag2_data_iteration = 0;
        h_warnings_static_warning_variable_total_sandbag3_data_iteration = 0;
        h_warnings_static_warning_variable_total_inform_others_data_iteration = 0;
        h_warnings_static_warning_variable_total_get_informed_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

float reduce_household_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->x),  thrust::device_pointer_cast(d_households_default->x) + h_xmachine_memory_household_default_count);
}

float min_household_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->y),  thrust::device_pointer_cast(d_households_default->y) + h_xmachine_memory_household_default_count);
}

float min_household_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_resident_num_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->resident_num),  thrust::device_pointer_cast(d_households_default->resident_num) + h_xmachine_memory_household_default_count);
}

int count_household_default_resident_num_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->resident_num),  thrust::device_pointer_cast(d_households_default->resident_num) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_resident_num_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->resident_num);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_resident_num_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->resident_num);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_OYI_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->OYI),  thrust::device_pointer_cast(d_households_default->OYI) + h_xmachine_memory_household_default_count);
}

int count_household_default_OYI_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->OYI),  thrust::device_pointer_cast(d_households_default->OYI) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_OYI_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->OYI);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_OYI_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->OYI);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_tenure_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->tenure),  thrust::device_pointer_cast(d_households_default->tenure) + h_xmachine_memory_household_default_count);
}

int count_household_default_tenure_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->tenure),  thrust::device_pointer_cast(d_households_default->tenure) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_tenure_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->tenure);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_tenure_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->tenure);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_take_measure_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->take_measure),  thrust::device_pointer_cast(d_households_default->take_measure) + h_xmachine_memory_household_default_count);
}

int count_household_default_take_measure_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->take_measure),  thrust::device_pointer_cast(d_households_default->take_measure) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_take_measure_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->take_measure);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_take_measure_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->take_measure);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_warning_area_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->warning_area),  thrust::device_pointer_cast(d_households_default->warning_area) + h_xmachine_memory_household_default_count);
}

int count_household_default_warning_area_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->warning_area),  thrust::device_pointer_cast(d_households_default->warning_area) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_warning_area_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->warning_area);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_warning_area_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->warning_area);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_get_warning_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->get_warning),  thrust::device_pointer_cast(d_households_default->get_warning) + h_xmachine_memory_household_default_count);
}

int count_household_default_get_warning_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->get_warning),  thrust::device_pointer_cast(d_households_default->get_warning) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_get_warning_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->get_warning);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_get_warning_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->get_warning);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_alert_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->alert_state),  thrust::device_pointer_cast(d_households_default->alert_state) + h_xmachine_memory_household_default_count);
}

int count_household_default_alert_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->alert_state),  thrust::device_pointer_cast(d_households_default->alert_state) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_alert_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->alert_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_alert_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->alert_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_sandbag_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->sandbag_state),  thrust::device_pointer_cast(d_households_default->sandbag_state) + h_xmachine_memory_household_default_count);
}

int count_household_default_sandbag_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->sandbag_state),  thrust::device_pointer_cast(d_households_default->sandbag_state) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_sandbag_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->sandbag_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_sandbag_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->sandbag_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_sandbag_time_count_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->sandbag_time_count),  thrust::device_pointer_cast(d_households_default->sandbag_time_count) + h_xmachine_memory_household_default_count);
}

int count_household_default_sandbag_time_count_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->sandbag_time_count),  thrust::device_pointer_cast(d_households_default->sandbag_time_count) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_sandbag_time_count_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->sandbag_time_count);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_sandbag_time_count_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->sandbag_time_count);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_flooded_time_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->flooded_time),  thrust::device_pointer_cast(d_households_default->flooded_time) + h_xmachine_memory_household_default_count);
}

int count_household_default_flooded_time_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->flooded_time),  thrust::device_pointer_cast(d_households_default->flooded_time) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_flooded_time_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->flooded_time);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_flooded_time_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->flooded_time);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_initial_wl_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->initial_wl),  thrust::device_pointer_cast(d_households_default->initial_wl) + h_xmachine_memory_household_default_count);
}

float min_household_default_initial_wl_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->initial_wl);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_initial_wl_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->initial_wl);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_actual_wl_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->actual_wl),  thrust::device_pointer_cast(d_households_default->actual_wl) + h_xmachine_memory_household_default_count);
}

float min_household_default_actual_wl_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->actual_wl);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_actual_wl_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->actual_wl);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_average_wl_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->average_wl),  thrust::device_pointer_cast(d_households_default->average_wl) + h_xmachine_memory_household_default_count);
}

float min_household_default_average_wl_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->average_wl);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_average_wl_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->average_wl);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_max_wl_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->max_wl),  thrust::device_pointer_cast(d_households_default->max_wl) + h_xmachine_memory_household_default_count);
}

float min_household_default_max_wl_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->max_wl);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_max_wl_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->max_wl);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_financial_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->financial_damage),  thrust::device_pointer_cast(d_households_default->financial_damage) + h_xmachine_memory_household_default_count);
}

float min_household_default_financial_damage_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->financial_damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_financial_damage_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->financial_damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_inform_others_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->inform_others),  thrust::device_pointer_cast(d_households_default->inform_others) + h_xmachine_memory_household_default_count);
}

int count_household_default_inform_others_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->inform_others),  thrust::device_pointer_cast(d_households_default->inform_others) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_inform_others_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->inform_others);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_inform_others_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->inform_others);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_get_informed_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->get_informed),  thrust::device_pointer_cast(d_households_default->get_informed) + h_xmachine_memory_household_default_count);
}

int count_household_default_get_informed_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->get_informed),  thrust::device_pointer_cast(d_households_default->get_informed) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_get_informed_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->get_informed);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_get_informed_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->get_informed);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_lod_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->lod),  thrust::device_pointer_cast(d_households_default->lod) + h_xmachine_memory_household_default_count);
}

int count_household_default_lod_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->lod),  thrust::device_pointer_cast(d_households_default->lod) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_lod_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->lod);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_lod_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->lod);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_household_default_animate_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->animate),  thrust::device_pointer_cast(d_households_default->animate) + h_xmachine_memory_household_default_count);
}

float min_household_default_animate_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->animate);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_household_default_animate_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_households_default->animate);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_household_default_animate_dir_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_households_default->animate_dir),  thrust::device_pointer_cast(d_households_default->animate_dir) + h_xmachine_memory_household_default_count);
}

int count_household_default_animate_dir_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_households_default->animate_dir),  thrust::device_pointer_cast(d_households_default->animate_dir) + h_xmachine_memory_household_default_count, count_value);
}
int min_household_default_animate_dir_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->animate_dir);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_household_default_animate_dir_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_households_default->animate_dir);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_household_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_flood_static_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_floods_static->x),  thrust::device_pointer_cast(d_floods_static->x) + h_xmachine_memory_flood_static_count);
}

int count_flood_static_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_floods_static->x),  thrust::device_pointer_cast(d_floods_static->x) + h_xmachine_memory_flood_static_count, count_value);
}
int min_flood_static_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_flood_static_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_flood_static_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_floods_static->y),  thrust::device_pointer_cast(d_floods_static->y) + h_xmachine_memory_flood_static_count);
}

int count_flood_static_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_floods_static->y),  thrust::device_pointer_cast(d_floods_static->y) + h_xmachine_memory_flood_static_count, count_value);
}
int min_flood_static_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_flood_static_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_flood_static_floodID_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_floods_static->floodID),  thrust::device_pointer_cast(d_floods_static->floodID) + h_xmachine_memory_flood_static_count);
}

int count_flood_static_floodID_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_floods_static->floodID),  thrust::device_pointer_cast(d_floods_static->floodID) + h_xmachine_memory_flood_static_count, count_value);
}
int min_flood_static_floodID_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->floodID);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_flood_static_floodID_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_floods_static->floodID);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_flood_static_flood_h_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_floods_static->flood_h),  thrust::device_pointer_cast(d_floods_static->flood_h) + h_xmachine_memory_flood_static_count);
}

float min_flood_static_flood_h_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_floods_static->flood_h);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_flood_static_flood_h_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_floods_static->flood_h);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_flood_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_warning_static_warning_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->x),  thrust::device_pointer_cast(d_warnings_static_warning->x) + h_xmachine_memory_warning_static_warning_count);
}

float min_warning_static_warning_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_warning_static_warning_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_warning_static_warning_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->y),  thrust::device_pointer_cast(d_warnings_static_warning->y) + h_xmachine_memory_warning_static_warning_count);
}

float min_warning_static_warning_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_warning_static_warning_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_flooded_households_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->flooded_households),  thrust::device_pointer_cast(d_warnings_static_warning->flooded_households) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_flooded_households_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->flooded_households),  thrust::device_pointer_cast(d_warnings_static_warning->flooded_households) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_flooded_households_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->flooded_households);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_flooded_households_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->flooded_households);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_warning_static_warning_total_financial_damage_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_financial_damage),  thrust::device_pointer_cast(d_warnings_static_warning->total_financial_damage) + h_xmachine_memory_warning_static_warning_count);
}

float min_warning_static_warning_total_financial_damage_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_financial_damage);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_warning_static_warning_total_financial_damage_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_financial_damage);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_take_measure_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure),  thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_take_measure_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure),  thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_take_measure_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_take_measure_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_take_measure);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_get_warning_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning),  thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_get_warning_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning),  thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_get_warning_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_get_warning_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_get_warning);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_alert_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state),  thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_alert_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state),  thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_alert_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_alert_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_alert_state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_sandbag1_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_sandbag1_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_sandbag1_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_sandbag1_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag1);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_sandbag2_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_sandbag2_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_sandbag2_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_sandbag2_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag2);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_sandbag3_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_sandbag3_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3),  thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_sandbag3_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_sandbag3_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_sandbag3);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_inform_others_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others),  thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_inform_others_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others),  thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_inform_others_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_inform_others_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_inform_others);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_warning_static_warning_total_get_informed_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed),  thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed) + h_xmachine_memory_warning_static_warning_count);
}

int count_warning_static_warning_total_get_informed_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed),  thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed) + h_xmachine_memory_warning_static_warning_count, count_value);
}
int min_warning_static_warning_total_get_informed_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_warning_static_warning_total_get_informed_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_warnings_static_warning->total_get_informed);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_warning_static_warning_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int household_output_financial_damage_infor_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** household_output_financial_damage_infor
 * Agent function prototype for output_financial_damage_infor function of household agent
 */
void household_output_financial_damage_infor(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_household_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_household_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_household_list* households_default_temp = d_households;
	d_households = d_households_default;
	d_households_default = households_default_temp;
	//set working count to current state count
	h_xmachine_memory_household_count = h_xmachine_memory_household_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_household_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_financial_damage_infor_count + h_xmachine_memory_household_count > xmachine_message_financial_damage_infor_MAX){
		printf("Error: Buffer size of financial_damage_infor message will be exceeded in function output_financial_damage_infor\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_financial_damage_infor, household_output_financial_damage_infor_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = household_output_financial_damage_infor_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_financial_damage_infor_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_financial_damage_infor_output_type, &h_message_financial_damage_infor_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_financial_damage_infor)
	//Reallocate   : false
	//Input        : 
	//Output       : financial_damage_infor
	//Agent Output : 
	GPUFLAME_output_financial_damage_infor<<<g, b, sm_size, stream>>>(d_households, d_financial_damage_infors);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_financial_damage_infor_count += h_xmachine_memory_household_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_financial_damage_infor_count, &h_message_financial_damage_infor_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_household_default_count+h_xmachine_memory_household_count > xmachine_memory_household_MAX){
		printf("Error: Buffer size of output_financial_damage_infor agents in state default will be exceeded moving working agents to next state in function output_financial_damage_infor\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  households_default_temp = d_households;
  d_households = d_households_default;
  d_households_default = households_default_temp;
        
	//update new state agent size
	h_xmachine_memory_household_default_count += h_xmachine_memory_household_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int household_identify_flood_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (blockSize * sizeof(xmachine_message_flood_cell));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** household_identify_flood
 * Agent function prototype for identify_flood function of household agent
 */
void household_identify_flood(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_household_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_household_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_household_list* households_default_temp = d_households;
	d_households = d_households_default;
	d_households_default = households_default_temp;
	//set working count to current state count
	h_xmachine_memory_household_count = h_xmachine_memory_household_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_household_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_identify_flood, household_identify_flood_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = household_identify_flood_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_flood_cell_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_x_byte_offset, tex_xmachine_message_flood_cell_x, d_flood_cells->x, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_x_offset = (int)tex_xmachine_message_flood_cell_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_x_offset, &h_tex_xmachine_message_flood_cell_x_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_y_byte_offset, tex_xmachine_message_flood_cell_y, d_flood_cells->y, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_y_offset = (int)tex_xmachine_message_flood_cell_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_y_offset, &h_tex_xmachine_message_flood_cell_y_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_floodID_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_floodID_byte_offset, tex_xmachine_message_flood_cell_floodID, d_flood_cells->floodID, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_floodID_offset = (int)tex_xmachine_message_flood_cell_floodID_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_floodID_offset, &h_tex_xmachine_message_flood_cell_floodID_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_flood_h_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_flood_h_byte_offset, tex_xmachine_message_flood_cell_flood_h, d_flood_cells->flood_h, sizeof(float)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_flood_h_offset = (int)tex_xmachine_message_flood_cell_flood_h_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_flood_h_offset, &h_tex_xmachine_message_flood_cell_flood_h_offset, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_household_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_household_scan_input<<<gridSize, blockSize, 0, stream>>>(d_households);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (identify_flood)
	//Reallocate   : true
	//Input        : flood_cell
	//Output       : 
	//Agent Output : 
	GPUFLAME_identify_flood<<<g, b, sm_size, stream>>>(d_households, d_flood_cells, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_floodID));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_flood_h));
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_household, 
        temp_scan_storage_bytes_household, 
        d_households->_scan_input,
        d_households->_position,
        h_xmachine_memory_household_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_household_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_household_Agents<<<gridSize, blockSize, 0, stream>>>(d_households_swap, d_households, 0, h_xmachine_memory_household_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_household_list* identify_flood_households_temp = d_households;
	d_households = d_households_swap;
	d_households_swap = identify_flood_households_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_households_swap->_position[h_xmachine_memory_household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_households_swap->_scan_input[h_xmachine_memory_household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_household_count = scan_last_sum+1;
	else
		h_xmachine_memory_household_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_household_default_count+h_xmachine_memory_household_count > xmachine_memory_household_MAX){
		printf("Error: Buffer size of identify_flood agents in state default will be exceeded moving working agents to next state in function identify_flood\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_household_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_household_Agents<<<gridSize, blockSize, 0, stream>>>(d_households_default, d_households, h_xmachine_memory_household_default_count, h_xmachine_memory_household_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_household_default_count += h_xmachine_memory_household_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int household_detect_flood_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (blockSize * sizeof(xmachine_message_flood_cell));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** household_detect_flood
 * Agent function prototype for detect_flood function of household agent
 */
void household_detect_flood(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_household_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_household_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_household_list* households_default_temp = d_households;
	d_households = d_households_default;
	d_households_default = households_default_temp;
	//set working count to current state count
	h_xmachine_memory_household_count = h_xmachine_memory_household_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_household_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_detect_flood, household_detect_flood_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = household_detect_flood_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_flood_cell_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_x_byte_offset, tex_xmachine_message_flood_cell_x, d_flood_cells->x, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_x_offset = (int)tex_xmachine_message_flood_cell_x_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_x_offset, &h_tex_xmachine_message_flood_cell_x_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_y_byte_offset, tex_xmachine_message_flood_cell_y, d_flood_cells->y, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_y_offset = (int)tex_xmachine_message_flood_cell_y_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_y_offset, &h_tex_xmachine_message_flood_cell_y_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_floodID_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_floodID_byte_offset, tex_xmachine_message_flood_cell_floodID, d_flood_cells->floodID, sizeof(int)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_floodID_offset = (int)tex_xmachine_message_flood_cell_floodID_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_floodID_offset, &h_tex_xmachine_message_flood_cell_floodID_offset, sizeof(int)));
	size_t tex_xmachine_message_flood_cell_flood_h_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_flood_cell_flood_h_byte_offset, tex_xmachine_message_flood_cell_flood_h, d_flood_cells->flood_h, sizeof(float)*xmachine_message_flood_cell_MAX));
	h_tex_xmachine_message_flood_cell_flood_h_offset = (int)tex_xmachine_message_flood_cell_flood_h_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_flood_cell_flood_h_offset, &h_tex_xmachine_message_flood_cell_flood_h_offset, sizeof(int)));
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_household_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_household_scan_input<<<gridSize, blockSize, 0, stream>>>(d_households);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (detect_flood)
	//Reallocate   : true
	//Input        : flood_cell
	//Output       : 
	//Agent Output : 
	GPUFLAME_detect_flood<<<g, b, sm_size, stream>>>(d_households, d_flood_cells, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_floodID));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_flood_cell_flood_h));
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_household, 
        temp_scan_storage_bytes_household, 
        d_households->_scan_input,
        d_households->_position,
        h_xmachine_memory_household_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_household_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_household_Agents<<<gridSize, blockSize, 0, stream>>>(d_households_swap, d_households, 0, h_xmachine_memory_household_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_household_list* detect_flood_households_temp = d_households;
	d_households = d_households_swap;
	d_households_swap = detect_flood_households_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_households_swap->_position[h_xmachine_memory_household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_households_swap->_scan_input[h_xmachine_memory_household_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_household_count = scan_last_sum+1;
	else
		h_xmachine_memory_household_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_household_default_count+h_xmachine_memory_household_count > xmachine_memory_household_MAX){
		printf("Error: Buffer size of detect_flood agents in state default will be exceeded moving working agents to next state in function detect_flood\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_household_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_household_Agents<<<gridSize, blockSize, 0, stream>>>(d_households_default, d_households, h_xmachine_memory_household_default_count, h_xmachine_memory_household_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_household_default_count += h_xmachine_memory_household_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int household_communicate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_state_data));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** household_communicate
 * Agent function prototype for communicate function of household agent
 */
void household_communicate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_household_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_household_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_household_list* households_default_temp = d_households;
	d_households = d_households_default;
	d_households_default = households_default_temp;
	//set working count to current state count
	h_xmachine_memory_household_count = h_xmachine_memory_household_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_count, &h_xmachine_memory_household_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_household_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_communicate, household_communicate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = household_communicate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (communicate)
	//Reallocate   : false
	//Input        : state_data
	//Output       : 
	//Agent Output : 
	GPUFLAME_communicate<<<g, b, sm_size, stream>>>(d_households, d_state_datas, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_household_default_count+h_xmachine_memory_household_count > xmachine_memory_household_MAX){
		printf("Error: Buffer size of communicate agents in state default will be exceeded moving working agents to next state in function communicate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  households_default_temp = d_households;
  d_households = d_households_default;
  d_households_default = households_default_temp;
        
	//update new state agent size
	h_xmachine_memory_household_default_count += h_xmachine_memory_household_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_household_default_count, &h_xmachine_memory_household_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int flood_output_flood_cells_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** flood_output_flood_cells
 * Agent function prototype for output_flood_cells function of flood agent
 */
void flood_output_flood_cells(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_flood_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_flood_static_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_flood_list* floods_static_temp = d_floods;
	d_floods = d_floods_static;
	d_floods_static = floods_static_temp;
	//set working count to current state count
	h_xmachine_memory_flood_count = h_xmachine_memory_flood_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_count, &h_xmachine_memory_flood_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_flood_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_flood_cells, flood_output_flood_cells_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = flood_output_flood_cells_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_flood_cells)
	//Reallocate   : false
	//Input        : 
	//Output       : flood_cell
	//Agent Output : 
	GPUFLAME_output_flood_cells<<<g, b, sm_size, stream>>>(d_floods, d_flood_cells);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	floods_static_temp = d_floods_static;
	d_floods_static = d_floods;
	d_floods = floods_static_temp;
    //set current state count
	h_xmachine_memory_flood_static_count = h_xmachine_memory_flood_count;

	//printf("flood_output_flood_cells --- %d and %d --- %p --- %p \n", d_xmachine_memory_navmap_static_count, h_xmachine_memory_navmap_static_count, &d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count);

	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));
	
	//printf("flood_output_flood_cells gpuErrchk finished \n");
	
	
	
}



	
/* Shared memory size calculator for agent function */
int flood_generate_warnings_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** flood_generate_warnings
 * Agent function prototype for generate_warnings function of flood agent
 */
void flood_generate_warnings(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_flood_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_flood_static_count;

	
	//FOR warning AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_warning_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_warning_scan_input<<<gridSize, blockSize, 0, stream>>>(d_warnings_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_flood_list* floods_static_temp = d_floods;
	d_floods = d_floods_static;
	d_floods_static = floods_static_temp;
	//set working count to current state count
	h_xmachine_memory_flood_count = h_xmachine_memory_flood_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_count, &h_xmachine_memory_flood_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_flood_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_warnings, flood_generate_warnings_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = flood_generate_warnings_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_warnings)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : warning
	GPUFLAME_generate_warnings<<<g, b, sm_size, stream>>>(d_floods, d_warnings_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE flood AGENTS ARE KILLED (needed for scatter)
	int floods_pre_death_count = h_xmachine_memory_flood_count;
	
	//FOR warning AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_warning, 
        temp_scan_storage_bytes_warning, 
        d_warnings_new->_scan_input, 
        d_warnings_new->_position, 
        floods_pre_death_count,
        stream
    );

	//reset agent count
	int warning_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_warnings_new->_position[floods_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_warnings_new->_scan_input[floods_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		warning_after_birth_count = h_xmachine_memory_warning_static_warning_count + scan_last_sum+1;
	else
		warning_after_birth_count = h_xmachine_memory_warning_static_warning_count + scan_last_sum;
	//check buffer is not exceeded
	if (warning_after_birth_count > xmachine_memory_warning_MAX){
		printf("Error: Buffer size of warning agents in state static_warning will be exceeded writing new agents in function generate_warnings\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_warning_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_warning_Agents<<<gridSize, blockSize, 0, stream>>>(d_warnings_static_warning, d_warnings_new, h_xmachine_memory_warning_static_warning_count, floods_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_warning_static_warning_count = warning_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	floods_static_temp = d_floods_static;
	d_floods_static = d_floods;
	d_floods = floods_static_temp;
    //set current state count
	h_xmachine_memory_flood_static_count = h_xmachine_memory_flood_count;

	//printf("flood_generate_warnings --- %d and %d --- %p --- %p \n", d_xmachine_memory_navmap_static_count, h_xmachine_memory_navmap_static_count, &d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count);

	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));
	
	//printf("flood_generate_warnings gpuErrchk finished \n");
	
	
	
}



	
/* Shared memory size calculator for agent function */
int flood_update_data_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** flood_update_data
 * Agent function prototype for update_data function of flood agent
 */
void flood_update_data(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_flood_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_flood_static_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_flood_list* floods_static_temp = d_floods;
	d_floods = d_floods_static;
	d_floods_static = floods_static_temp;
	//set working count to current state count
	h_xmachine_memory_flood_count = h_xmachine_memory_flood_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_count, &h_xmachine_memory_flood_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_flood_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_update_data, flood_update_data_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = flood_update_data_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (update_data)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_update_data<<<g, b, sm_size, stream>>>(d_floods, FLOOD_DATA_ARRAY);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	floods_static_temp = d_floods_static;
	d_floods_static = d_floods;
	d_floods = floods_static_temp;
    //set current state count
	h_xmachine_memory_flood_static_count = h_xmachine_memory_flood_count;

	//printf("flood_update_data --- %d and %d --- %p --- %p \n", d_xmachine_memory_navmap_static_count, h_xmachine_memory_navmap_static_count, &d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count);

	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_flood_static_count, &h_xmachine_memory_flood_static_count, sizeof(int)));
	
	//printf("flood_update_data gpuErrchk finished \n");
	
	
	
}



	
/* Shared memory size calculator for agent function */
int warning_calcu_damage_infor_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_financial_damage_infor));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** warning_calcu_damage_infor
 * Agent function prototype for calcu_damage_infor function of warning agent
 */
void warning_calcu_damage_infor(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_warning_static_warning_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_warning_static_warning_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_warning_list* warnings_static_warning_temp = d_warnings;
	d_warnings = d_warnings_static_warning;
	d_warnings_static_warning = warnings_static_warning_temp;
	//set working count to current state count
	h_xmachine_memory_warning_count = h_xmachine_memory_warning_static_warning_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_count, &h_xmachine_memory_warning_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_warning_static_warning_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_calcu_damage_infor, warning_calcu_damage_infor_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = warning_calcu_damage_infor_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (calcu_damage_infor)
	//Reallocate   : false
	//Input        : financial_damage_infor
	//Output       : 
	//Agent Output : 
	GPUFLAME_calcu_damage_infor<<<g, b, sm_size, stream>>>(d_warnings, d_financial_damage_infors);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_warning_static_warning_count+h_xmachine_memory_warning_count > xmachine_memory_warning_MAX){
		printf("Error: Buffer size of calcu_damage_infor agents in state static_warning will be exceeded moving working agents to next state in function calcu_damage_infor\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  warnings_static_warning_temp = d_warnings;
  d_warnings = d_warnings_static_warning;
  d_warnings_static_warning = warnings_static_warning_temp;
        
	//update new state agent size
	h_xmachine_memory_warning_static_warning_count += h_xmachine_memory_warning_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int warning_output_state_data_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** warning_output_state_data
 * Agent function prototype for output_state_data function of warning agent
 */
void warning_output_state_data(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_warning_static_warning_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_warning_static_warning_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//init_flood_data_array();
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_warning_list* warnings_static_warning_temp = d_warnings;
	d_warnings = d_warnings_static_warning;
	d_warnings_static_warning = warnings_static_warning_temp;
	//set working count to current state count
	h_xmachine_memory_warning_count = h_xmachine_memory_warning_static_warning_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_count, &h_xmachine_memory_warning_count, sizeof(int)));	
	//set current state count to 0
	
	
	h_xmachine_memory_warning_static_warning_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_state_data_count + h_xmachine_memory_warning_count > xmachine_message_state_data_MAX){
		printf("Error: Buffer size of state_data message will be exceeded in function output_state_data\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_state_data, warning_output_state_data_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = warning_output_state_data_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_state_data_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_state_data_output_type, &h_message_state_data_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_state_data)
	//Reallocate   : false
	//Input        : 
	//Output       : state_data
	//Agent Output : 
	GPUFLAME_output_state_data<<<g, b, sm_size, stream>>>(d_warnings, d_state_datas);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_state_data_count += h_xmachine_memory_warning_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_state_data_count, &h_message_state_data_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_warning_static_warning_count+h_xmachine_memory_warning_count > xmachine_memory_warning_MAX){
		printf("Error: Buffer size of output_state_data agents in state static_warning will be exceeded moving working agents to next state in function output_state_data\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  warnings_static_warning_temp = d_warnings;
  d_warnings = d_warnings_static_warning;
  d_warnings_static_warning = warnings_static_warning_temp;
        
	//update new state agent size
	h_xmachine_memory_warning_static_warning_count += h_xmachine_memory_warning_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_warning_static_warning_count, &h_xmachine_memory_warning_static_warning_count, sizeof(int)));	
	
	
}


 
extern void reset_household_default_count()
{
    h_xmachine_memory_household_default_count = 0;
}
 
extern void reset_flood_static_count()
{
    h_xmachine_memory_flood_static_count = 0;
}
 
extern void reset_warning_static_warning_count()
{
    h_xmachine_memory_warning_static_warning_count = 0;
}
