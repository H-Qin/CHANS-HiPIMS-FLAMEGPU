
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
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#ifdef VISUALISATION
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#include "header.h"

#if defined(PROFILE)
unsigned int g_profile_colour_id = 0;
#endif

/* IO Variables*/
char inputfile[100];          /**< Input path char buffer*/
char outputpath[1000];         /**< Output path char buffer*/

// Define the default value indicating if XML output should be produced or not.
#define OUTPUT_TO_XML 3600

#define HELP_OPTION_SHORT "-h"
#define HELP_OPTION_LONG "--help"

/** checkUsage
 * Function to check the correct number of arguments
 * @param arc	main argument count
 * @param argv	main argument values
 * @return true if usage is correct, otherwise false
 */
int checkUsage(int argc, char** argv) {
	// Initalise return value.
	int retval = true;

	// Get the EXE name.
	char * executable = nullptr;
	size_t i = 0;
	size_t last = 0;
	while (argv[0][i] != '\0')
	{
		/* For windows directories */
		if (argv[0][i] == '\\') last = i + 1;
		/* For unix directories */
		if (argv[0][i] == '/') last = i + 1;
		i++;
	}

	size_t substrLen = strlen(argv[0]) - last;
	executable = (char*)malloc(substrLen + 1);
	if (executable != nullptr) {
		executable[substrLen] = '\0';
		strncpy(executable, argv[0] + last, substrLen);
	}

	// Iterate each argument, looking for the help flag.
	bool helpFlagFound = false;
	for(int index = 1; index < argc; index++){
		if(strcmp(HELP_OPTION_SHORT, argv[index]) == 0 || strcmp(HELP_OPTION_LONG, argv[index]) == 0){
			helpFlagFound = true;
			break;
		}
	}

	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if(helpFlagFound || argc < 2 || argc > 3)
	{
		printf("\nusage: %s [-h] [--help] input_path [cuda_device_id]\n", executable != nullptr ? executable : "main");
		printf("\n");
		printf("required arguments:\n");
		printf("  input_path           Path to initial states XML file OR path to output XML directory\n");
		printf("\n");
		printf("options arguments:\n");
		printf("  -h, --help           Output this help message.\n");
		printf("  cuda_device_id       CUDA device ID to be used. Default is 0\n");
		// Set the appropriate return value
		retval = false;
	}
#else
	printf("FLAMEGPU Console mode\n");
	if(helpFlagFound || argc < 3 || argc > 5)
	{
		printf("\nusage: %s [-h] [--help] input_path num_iterations [cuda_device_id] [XML_output_override]\n", executable != nullptr ? executable : "main");
		printf("\n");
		printf("required arguments:\n");
		printf("  input_path           Path to initial states XML file OR path to output XML directory\n");
		printf("  num_iterations       Number of simulation iterations\n");
		printf("\n");
		printf("options arguments:\n");
		printf("  -h, --help           Output this help message.\n");
		printf("  cuda_device_id       CUDA device ID to be used. Default is 0.\n");
		printf("  XML_output_frequency Frequency of XML output\n");
		printf("                         0 = No output\n");
		printf("                         1 = Every 1 iteration\n");
		printf("                         5 = Every 5 iterations\n");
		printf("                         Default value: %d\n", OUTPUT_TO_XML);
		// Set the appropriate return value
		retval = false;
	}
#endif

	// Free malloced memory
	free(executable);
	executable = nullptr;
	// return the appropriate code.
	return retval;
}

/** getOutputDir
 * Function which gets the global char array contianign the path for output
 * @return char array containing relative path to output locaiton
 */
const char* getOutputDir(){
    return outputpath;
}


/** parentDirectoryOfPath
* Function which given a path removes the last segment, copying into a pre-defined buffer.
* @param parent pre allocated buffer for the shoretened path
* @param path input path to be shortented
*/
void parentDirectoryOfPath(char * parent, char * path) {
	int i = 0;
	int lastd = -1;
	while (path[i] != '\0')
	{
		/* For windows directories */
		if (path[i] == '\\') lastd = i;
		/* For unix directories */
		if (path[i] == '/') lastd = i;
		i++;
	}
	strcpy(parent, path);
	//parent[lastd + 1] = '\0';
	// Replace the traling slash, as files and directories cannot have the same name.
	parent[lastd + 1] = '\0';
}

/** getPathProperties
* Function to get information about a filepath, if it exists, is a file or is a directory
* @param path path to be checked
* @param isFile returned boolean indicating if the path points to a file.
* @param isDir return boolean indicating if the path points to a directory.
* @return boolean indicating if the path exists.
*/
bool getPathProperties(char * path, bool * isFile, bool * isDir) {
	bool fileExists = false;
	// Initialse bools to false.
	*isFile = false;
	*isDir = false;

	// Buffer for stat output.
	struct stat statBuf {0};
	// Use stat to query the path information.
	int statResult = stat(path, &statBuf);

	// If stat was successfull
	if (statResult == 0) {
		// Update return values indicating if the path is a file or a directory.
		*isDir = (statBuf.st_mode & S_IFDIR) != 0;
		*isFile = (statBuf.st_mode & S_IFREG) != 0;
		fileExists = *isDir || *isFile;
	} 
	// Otherwise if stat did report an errr.
	else {
		// If the file does not exist, set this and continue.
		if (errno == ENOENT) {
			fileExists = false;
		}
		// For any other errors, we should abort.
		else {
			fprintf(stderr, "Error: An unknown error occured while processing file infomration.\n");
			fflush(stdout);
			exit(EXIT_FAILURE);
		}
	}
	// Return if the file exists or not.
	return fileExists;
}

/** setFilePaths
 * Function to set global variables for the input XML file and its directory location
 *@param input input path of model xml file
 */
void setFilePaths(char* input){
	PROFILE_SCOPED_RANGE("setFilePaths");

	// Get infomration about the inputpath file.
	bool inputIsFile = false;
	bool inputIsDir = false;
	bool inputExists = getPathProperties(input, &inputIsFile, &inputIsDir);

	// If input exists:
	if (inputExists) {
		// If it is a file
		if (inputIsFile) {
			//Copy input file, and proceed as normal.
			strcpy(inputfile, input);
			// We must get the parent directory as the output directory.
			parentDirectoryOfPath(outputpath, inputfile);
		}
		// Otherwise it is a directory
		else {
			// We do not have an input file., but use this as the directory.
			inputfile[0] = '\0';
			strcpy(outputpath, input);
		}
	}
	// Otherwise if the input file does not exist
	else {
		// The input path is empty.
		inputfile[0] = '\0';
		// Try to find a parent directory.
		parentDirectoryOfPath(outputpath, input);

		// Check if the parent directory exists.
		bool dirIsFile = false;
		bool dirIsDir = false;
		bool dirExists = getPathProperties(outputpath, &dirIsFile, &dirIsDir);

		// If the dir exists
		if (dirExists) {
			// IF the dir is not a directory, it is a file. Abort.
			if (!dirIsDir || dirIsFile) {
				printf("Error: outputpath `%s` exists, but it is not a directory.\n", outputpath);
				exit(EXIT_FAILURE);
			}
			else {
				// Otherwise the parent directory exists and is a directory.
				printf("Warning: `%s` does not exist using parent directory for output.\n", input);
			}
		}
		else {
			// If the directory does not exist, use the working directory.
			printf("Warning: Parent directory `%s` does not exist. Using current working directory for output.\n", outputpath);
			outputpath[0] = '\0';
		}
	}

	printf("Initial states: %s\n", inputfile[0] != '\0' ? inputfile : "(none)");
	printf("Output dir: %s\n", outputpath[0] != '\0' ? outputpath : "(cwd)");
}


int getOutputXMLFrequency(int argc, char**argv){

#ifdef VISUALISATION
	// If visualisation mode is set, we do not output.
	return 0;
#else
	// Initialise to #defined default
	int outputFrequency = OUTPUT_TO_XML;
	// If console mode is set and we have the right number of arguments, use the relevant index.
	if (argc >= 5){
		outputFrequency = (int) atoi(argv[4]);
		if(outputFrequency <= 0){
			outputFrequency = 0;
		}
	}
	return outputFrequency;
#endif

}

void initCUDA(int argc, char** argv){
	PROFILE_SCOPED_RANGE("initCUDA");
	cudaError_t cudaStatus;
	int device;
	int device_count;

	//default device
	device = 0;
	cudaStatus = cudaGetDeviceCount(&device_count);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?\n");
		exit(EXIT_FAILURE);
	}
	if (device_count == 0){
		fprintf(stderr, "Error no CUDA devices found!\n");
		exit(EXIT_FAILURE);
	}

#ifdef VISUALISATION
	if (argc >= 3){
		device = atoi(argv[2]);
	}
#else
	if (argc >= 4){
		device = atoi(argv[3]);
	}
#endif

	if (device >= device_count){
		fprintf(stderr, "Error selecting CUDA device! Device id '%d' is not found?\n", device);
		exit(EXIT_FAILURE);
	}

	// Select device
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error setting CUDA device!\n");
		exit(EXIT_FAILURE);
	}
	// Get device properties.
	cudaDeviceProp props;
	cudaStatus = cudaGetDeviceProperties(&props, device);
	if(cudaStatus == cudaSuccess){
	#ifdef _MSC_VER
		const char * driverMode = props.tccDriver ? "TCC" : "WDDM";
	#else
		const char * driverMode = "Linux";
	#endif
		fprintf(stdout, "GPU %d: %s, SM%d%d, %s, pciBusId %d\n", device, props.name, props.major, props.minor, driverMode, props.pciBusID);
	} else {
		fprintf(stderr, "Error Accessing Cuda Device properties for GPU %d\n", device);
	}

	cudaFree(0);

}

void runConsoleWithoutXMLOutput(int iterations){
	PROFILE_SCOPED_RANGE("runConsoleWithoutXMLOutput");
	// Iteratively tun the correct number of iterations.
	for (int i=0; i< iterations; i++)
	{
		printf("Processing Simulation Step %i\n", i+1);
		//single simulation iteration
		singleIteration();
	}
}

void runConsoleWithXMLOutput(int iterations, int outputFrequency){
	PROFILE_SCOPED_RANGE("runConsoleWithXMLOutput");
	// Iteratively tun the correct number of iterations.
	for (int i=0; i< iterations; i++)
	{
		printf("Processing Simulation Step %i\n", i+1);
		//single simulation iteration
		singleIteration();
		// Save the iteration data to disk
		if((i+1) % outputFrequency == 0){
			saveIterationData(outputpath, i+1, get_host_household_default_agents(), get_device_household_default_agents(), get_agent_household_default_count(),get_host_flood_static_agents(), get_device_flood_static_agents(), get_agent_flood_static_count(),get_host_warning_static_warning_agents(), get_device_warning_static_warning_agents(), get_agent_warning_static_warning_count());
			printf("Iteration %i Saved to XML\n", i+1);
		}
	}

	// If we did not yet output the final iteration, output the final iteration.
	if(iterations % outputFrequency != 0){
		saveIterationData(outputpath, iterations, get_host_household_default_agents(), get_device_household_default_agents(), get_agent_household_default_count(),get_host_flood_static_agents(), get_device_flood_static_agents(), get_agent_flood_static_count(),get_host_warning_static_warning_agents(), get_device_warning_static_warning_agents(), get_agent_warning_static_warning_count());
		printf("Iteration %i Saved to XML\n", iterations);
	}

}

/**
 * Program main (Handles arguments)
 */
int main( int argc, char** argv) 
{
	cudaError_t cudaStatus;
	//check usage mode
	if (!checkUsage(argc, argv))
		exit(EXIT_FAILURE);

	//get the directory paths
	setFilePaths(argv[1]);

	//determine frequency we want to output to xml.
	int outputXMLFrequency = getOutputXMLFrequency(argc, argv);

	//initialise CUDA
	initCUDA(argc, argv);

#ifdef VISUALISATION
	//Init visualisation must be done before simulation init
	initVisualisation();
#endif

	//initialise the simulation
	initialise(inputfile);

    
#ifdef VISUALISATION
	runVisualisation();
	exit(EXIT_SUCCESS);
#else	
	//Benchmark simulation
	cudaEvent_t start, stop;
	float milliseconds = 0;
	
	//create timing events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//Get the number of iterations
	int iterations = atoi(argv[2]);
	if (iterations <= 0)
	{
		printf("Second argument must be a positive integer (Number of Iterations)\n");
		exit(EXIT_FAILURE);
	}
  
	//start timing
	cudaEventRecord(start);

	// Launch the main loop with / without xml output.
	if(outputXMLFrequency > 0){
		runConsoleWithXMLOutput(iterations, outputXMLFrequency);
	} else {
		runConsoleWithoutXMLOutput(iterations);	
	}
	

	//CUDA stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf( "Total Processing time: %f (ms)\n", milliseconds);
#endif

	cleanup();
	PROFILE_PUSH_RANGE("cudaDeviceReset");
	cudaStatus = cudaDeviceReset();
	PROFILE_POP_RANGE();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error resetting the device!\n");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
