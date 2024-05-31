 /*
 * FLAMEGPU (Flexible Large-scale Agent Modelling Environment for Graphics Processing Units)
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

/*
 * Modifications and enhancements made by Haoyang Qin, 2024.
 * Loughborough University
 * Contact: h.qin@lboro.ac.uk
 *
 * This file is part of the HiPIMS-FLAMEGPU Coupled Model Framework.
 * All modifications and additions are the property of Haoyang Qin.
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * Haoyang Qin is strictly prohibited.
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include "header.h"
#include <iostream>
#include "rpc/client.h"
#include <string>
#include <vector>
#include "CustomVisualisation.h"
using std::vector;

#define SCALE_FACTOR 0.03125
#define I_SCALER (SCALE_FACTOR*0.035f)
#define NUM_FLOOD 900000
#define TIME_STEP 120

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

__FLAME_GPU_INIT_FUNC__ void initConstants(){

	// allocate address for FLOOD_DATA_ARRAY
	if (FLOOD_DATA_ARRAY == nullptr) {
		printf("TRY TO ALLOC address \n");
		cudaMalloc((void**)&FLOOD_DATA_ARRAY, NUM_FLOOD * sizeof(short));
		printf("Address of flood data arr at %p \n", FLOOD_DATA_ARRAY);
	}
	freopen("Results.txt", "w", stdout);
}

__FLAME_GPU_STEP_FUNC__ void read_data_func() {

	// update TIME
	const int* getTime = NULL;
	getTime = get_TIME();
	int changeTime = 0;
	changeTime = *getTime;
	changeTime += 1;
	set_TIME(&changeTime);

	// generate RANDOM SEED
	time_t timetemp;
	struct tm* p;
	time(&timetemp);
	p = localtime(&timetemp);
	int RandomSeed_SEC = 0;
	int RandomSeed_MIN = 0;
	RandomSeed_SEC = p->tm_sec;
	RandomSeed_MIN = p->tm_min;
	set_RANDOM_SEED_SEC(&RandomSeed_SEC);
	set_RANDOM_SEED_MIN(&RandomSeed_MIN);

	if (int(changeTime % TIME_STEP) == 0) {

		rpc::client c("127.0.0.1", 6060);
		auto result = c.call("request_data", int(changeTime)).as<std::vector<float>>();
		std::vector<short> short_array;

		for (int it = 0; it < result.size(); it++) {
			short_array.push_back(short(result[it] * 1000));
		}
		auto* FloodArray = &short_array[0];

		// SET NEXT TARGET
		c.call("set_target", int(changeTime) + TIME_STEP);
		set_FLOOD_DATA_ARRAY(FloodArray);
	}
}

__FLAME_GPU_FUNC__ int output_flood_cells(xmachine_memory_flood* agent, xmachine_message_flood_cell_list* flood_cell_messages){
	add_flood_cell_message<DISCRETE_2D>(flood_cell_messages, 
		agent->x, agent->y, 
		agent->floodID,
		agent->flood_h);
    return 0;
}

__FLAME_GPU_FUNC__ int output_financial_damage_infor(xmachine_memory_household* agent, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages) {
	add_financial_damage_infor_message(financial_damage_infor_messages, agent->x, agent->y, 0.0, agent->max_wl,
		agent->financial_damage, agent->take_measure, agent->get_warning, agent->alert_state, agent->sandbag_state, agent->inform_others, agent->get_informed, 
		agent->flooded_time, agent->actual_wl);
	return 0;
}

__FLAME_GPU_FUNC__ int output_state_data(xmachine_memory_warning* agent, xmachine_message_state_data_list* state_data_messages) {

	add_state_data_message(state_data_messages, agent->x, agent->y, agent->flooded_households,
		agent->total_financial_damage, agent->total_take_measure, agent->total_get_warning, agent->total_alert_state,
		agent->total_sandbag1, agent->total_sandbag2, agent->total_sandbag3,
		agent->total_inform_others, agent->total_get_informed);

	return 0;
}

__FLAME_GPU_FUNC__ int update_data(xmachine_memory_flood* agent, short* data) {
	if (int(TIME % TIME_STEP) == 0) {
		if (agent->floodID > 0) {
			agent->flood_h = float(data[agent->floodID] / 1000.0);
		}
		else {
			agent->flood_h = 0.0f;
		}
	}
	return 0;
}

__FLAME_GPU_FUNC__ int calcu_damage_infor(xmachine_memory_warning* agent, xmachine_message_financial_damage_infor_list* financial_damage_infor_messages) {

	if (int(TIME % 300) == 0) {
		int count = 0;
		int total_in_alert_state = 0;
		int total_after_alert_state = 0;
		int total_sandbag4 = 0;
		xmachine_message_financial_damage_infor* current_message = get_first_financial_damage_infor_message(financial_damage_infor_messages);
		agent->total_financial_damage = 0;
		agent->total_take_measure = 0;
		agent->total_get_warning = 0;
		agent->total_sandbag1 = 0;
		agent->total_sandbag2 = 0;
		agent->total_sandbag3 = 0;
		agent->total_inform_others = 0;
		agent->total_get_informed = 0;
		agent->flooded_households = 0;

		while (current_message) {
			agent->total_financial_damage += current_message->financial_damage;
			agent->total_take_measure += current_message->take_measure;
			agent->total_get_warning += current_message->get_warning;
			agent->total_inform_others += current_message->inform_others;
			agent->total_get_informed += current_message->get_informed;
			if (current_message->max_wl > 0.3) {
				agent->flooded_households += 1;
			}
			if (current_message->alert_state == 1) {
				total_in_alert_state += 1;
			}
			else if (current_message->alert_state == 2) {
				total_after_alert_state += 1;
			}
			if (current_message->sandbag_state == 1) {
				agent->total_sandbag1 += 1;
			}
			else if (current_message->sandbag_state == 2) {
				agent->total_sandbag2 += 1;
			}
			else if (current_message->sandbag_state == 3) {
				agent->total_sandbag3 += 1;
			}
			else if (current_message->sandbag_state == 4) {
				total_sandbag4 += 1;
			}
			current_message = get_next_financial_damage_infor_message(current_message, financial_damage_infor_messages);
		}
		
		// for output
		printf("%d %d %f %d %d %d %d %d %d %d %d\n", TIME, agent->flooded_households, agent->total_financial_damage, agent->total_take_measure, total_in_alert_state, total_after_alert_state,
			agent->total_sandbag1, agent->total_sandbag2, agent->total_sandbag3, total_sandbag4, agent->total_inform_others);
		//printf("%d\n", agent->total_take_measure);

		//printf("total properties: %d , flooded households: %d,total financial damage: %f\n", count, agent->flooded_households, agent->total_fina_damage);
		//printf("total take measure: %d, total get warning: %d, total alert state: %d\n", agent->total_take_measure, agent->total_get_warning, agent->total_alert_state);
		//printf("apply sandbag: %d, effective sandbag: %d, useless sandbag: %d\n", agent->total_sandbag1, agent->total_sandbag2, agent->total_sandbag3);
		//printf("total inform others: %d, total get informed: %d\n", agent->total_inform_others, agent->total_get_informed);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int identify_flood(xmachine_memory_household* agent, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48) {

	int iter_num = RANDOM_SEED_SEC + 5;
	for (int i = 0; i < iter_num; i++) {
		float random_pre = rnd<CONTINUOUS>(rand48);
	}

	// identify whether the household responds to the warning
	if (int(TIME) == TIME_STEP) {
		// 95% properties respond to the warning
		float respond_ratio = rnd<CONTINUOUS>(rand48);
		if (respond_ratio < 0.95) {
			// all old & young & ill (OYI) people in the household
			if (agent->OYI == agent->resident_num) {
				float all_OYI_res_ratio = rnd<CONTINUOUS>(rand48);
				if (all_OYI_res_ratio < 0.95) {
					agent->take_measure = 0;
				}
				else { // all OYI don't use sandbags
					agent->take_measure = 1;
				}
			}
			// living alone
			else if (agent->resident_num == 1) {
				float alone_res_ratio = rnd<CONTINUOUS>(rand48);
				if (alone_res_ratio < 0.4) {
					agent->take_measure = 0; // no response
				}
				else {
					agent->take_measure = 1; // take response
				}
			}
			// parts of people are OYI
			else if (agent->OYI > 0) {
				float part_OYI_res_ratio = rnd<CONTINUOUS>(rand48);
				if (part_OYI_res_ratio < 0.3) {
					agent->take_measure = 0; // no response
				}
				else {
					agent->take_measure = 1; // take response
				}
			}
			// renting properties
			else if (agent->tenure == 1) {
				float rent_res_ratio = rnd<CONTINUOUS>(rand48);
				if (rent_res_ratio < 0.2) {
					agent->take_measure = 0; // no response
				}
				else {
					agent->take_measure = 1; // take response
				}
			}
			else {
				agent->take_measure = 1; // take response
			}
		}
		else {
			agent->take_measure = 0;
		}
	}

	// All respond scenario
	//if (int(TIME) == TIME_STEP) {
	//	agent->take_measure = 1;
	//}

	// Warning at 15:28 on 5th Dec
	//if (int(TIME) == 66480) {
	//	// #1 baseline
	//	// 89% properties are issued by warning system
	//	float warning_ratio = rnd<CONTINUOUS>(rand48);
	//	// 70% properties successfully receive the warning
	//	float receive_ratio = rnd<CONTINUOUS>(rand48);
	//	if (agent->warning_area == 1 && warning_ratio < 0.89 && receive_ratio < 0.7) {
	//		agent->get_warning = 1;
	//	}

	//	//// #3 all warning
	//	//if (agent->warning_area == 1) {
	//	//	agent->get_warning = 1;
	//	//}
	//}

	// initiate the water depth
	if (int(TIME) == TIME_STEP) {
		int x = floor(((agent->x + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
		int y = floor(((agent->y + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
		xmachine_message_flood_cell* current_message = get_first_flood_cell_message<CONTINUOUS>(flood_cell_messages, x, y);
		agent->initial_wl = current_message->flood_h;
	}

	// update max water depth, average water depth
	if (int(TIME % TIME_STEP) == 0) {
		//map agent position into 2d grid
		int x = floor(((agent->x + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
		int y = floor(((agent->y + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
		xmachine_message_flood_cell* current_message = get_first_flood_cell_message<CONTINUOUS>(flood_cell_messages, x, y);
		agent->actual_wl = current_message->flood_h - agent->initial_wl;
		if (agent->max_wl < agent->actual_wl) {
			agent->max_wl = agent->actual_wl;
		}

		// calculate financial damage of households
		if (agent->actual_wl > 0.25) {
			float wl_total = agent->flooded_time * agent->average_wl;
			agent->flooded_time = agent->flooded_time + 1; // the property is flooded in this time period (TIME_STEP)
			wl_total = wl_total + agent->actual_wl;
			agent->average_wl = wl_total / agent->flooded_time;

			float logH = log(agent->max_wl);
			double Ho = agent->average_wl * 1000; // H_out unit: mm
			float Hi = Ho - pow((pow(Ho,-0.86) + 0.000000516 * agent->flooded_time * TIME_STEP), -1.16279); // H_in unit: mm
			float logH_sandbag = log(Hi / 1000);
			float a1 = 11616.4; // abcd1 for no response
			float b1 = 4030.7;
			float c1 = 969.4;
			float d1 = 30139.1;
			float a2 = 11362.8; // abcd2 for taking response
			float b2 = 4169.7;
			float c2 = 998.7;
			float d2 = 28626.8;

			if (agent->take_measure == 1) {
				if (agent->sandbag_state == 10) { // 2 for effective sandbagging, 10 for no sandbagging
					agent->financial_damage = (a2 * logH_sandbag) + (b2 * logH_sandbag * logH_sandbag) + (c2 * logH_sandbag * logH_sandbag * logH_sandbag) + d2;
				}
				else {
					agent->financial_damage = (a2 * logH) + (b2 * logH * logH) + (c2 * logH * logH * logH) + d2;
				}
			}
			else {
				agent->financial_damage = (a1 * logH) + (b1 * logH * logH) + (c1 * logH * logH * logH) + d1;
			}
		}
	}
	
	// sandbag application and installation
	if (int(TIME % TIME_STEP) == 0) {
		if (agent->max_wl > 0.05 && agent->take_measure == 1 && agent->sandbag_time_count == 0 && 
			agent->sandbag_state != 2 && agent->sandbag_state != 3 && agent->sandbag_state != 4) {
			float sandbag_ratio = rnd<CONTINUOUS>(rand48);

			// probability of needing sandbags increases with water depth
			float probability = (1.73 * agent->max_wl - 0.04) * 0.0008;
			if (agent->max_wl < 0.6) {
				if (sandbag_ratio < probability) {
					agent->sandbag_state = 1; // sandbag_state 1 is applying and installing sandbags
					agent->sandbag_time_count = 9000; // 2.5 hrs to apply, get, and install sandbags
				}
			}
			else {
				agent->sandbag_state = 1;
				agent->sandbag_time_count = 9000; // 2.5 hrs to apply, get, and install sandbags
			}
		}
		if (agent->sandbag_state == 1) {
			agent->sandbag_time_count -= TIME_STEP;
			if (agent->sandbag_time_count < 1) {
				agent->sandbag_state = 2; // sandbag_state 2 is finished installing sandbags
				if (agent->max_wl > 0.3) {
					agent->sandbag_state = 3; // sandbag_state 3 is useless with sandbags
				}
			}
		}
		if (agent->sandbag_state == 2 && agent->actual_wl > 0.3) {
			agent->sandbag_state = 4;
		}
	}

	return 0;
}

__FLAME_GPU_FUNC__ int detect_flood(xmachine_memory_household* agent, xmachine_message_flood_cell_list* flood_cell_messages, RNG_rand48* rand48) {
	// random seed
	int iter_num = RANDOM_SEED_SEC + 5;
	for (int i = 0; i < iter_num; i++) {
		float random_pre = rnd<CONTINUOUS>(rand48);
	}

	int x = floor(((agent->x + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
	int y = floor(((agent->y + ENV_MAX) / ENV_WIDTH) * d_message_flood_cell_width);
	xmachine_message_flood_cell* current_message = get_first_flood_cell_message<CONTINUOUS>(flood_cell_messages, x, y);
	float acutal_water_depth = current_message->flood_h - agent->initial_wl;

	// whether change into the alert_state
	if (int(TIME % 300) == 0 && agent->take_measure == 0 && agent->resident_num != agent->OYI) {
		if (agent->get_warning == 1) {
			agent->alert_state = 1;
		}
		else {
			float detect_ratio = rnd<CONTINUOUS>(rand48);
			if (detect_ratio < 0.0023) { // observe the surroundings at regular intervals
				if (acutal_water_depth > 0.1) { // detect the water level around > 0.1m
					agent->alert_state = 1;
				}
			}
		}
	}

	// in the alert_state, check surrounding water depth more frequently
	if (int(TIME % 300) == 0 && agent->alert_state == 1) { 
		float detect_ratio = rnd<CONTINUOUS>(rand48);
		if (detect_ratio < 0.0833) { // observe the surroundings at more frequent intervals
			if (acutal_water_depth > 0.3) { // detect the water level around > 0.3 m
				float take_measure_ratio = rnd<CONTINUOUS>(rand48);
				if (take_measure_ratio < 0.9) { // 90% will change to take measure
					agent->take_measure = 1;
					agent->alert_state = 2; // change the state of the household
				}
			}
		}
	}
	return 0;
}

__FLAME_GPU_FUNC__ int communicate(xmachine_memory_household* agent, xmachine_message_state_data_list* state_data_messages, RNG_rand48* rand48) {

	// random seed
	int iter_num = RANDOM_SEED_SEC + 5;
	for (int i = 0; i < iter_num; i++) {
		float random_pre = rnd<CONTINUOUS>(rand48);
	}

	xmachine_message_state_data* current_message = get_first_state_data_message(state_data_messages);

	// flooded households inform others
	if (int(TIME % TIME_STEP) == 0 && agent->max_wl > 0.3 && agent->inform_others == 0 && agent->take_measure == 1) {
		float inform_ratio = rnd<CONTINUOUS>(rand48) * 10;
		agent->inform_others = ceil(inform_ratio / 2); // The num (1-5) of other households to be noticed
	}
	float to_inform_ratio = ((current_message->total_inform_others - current_message->total_get_informed)/5.0) / 40627.0;
	if (int(TIME % TIME_STEP) == 0 && to_inform_ratio > 0.0) {
		float get_informed_ratio = rnd<CONTINUOUS>(rand48);
		if (get_informed_ratio < to_inform_ratio && agent->resident_num != agent->OYI) {
			float get_informed_then_do_ratio = rnd<CONTINUOUS>(rand48);
			if (agent->alert_state == 0 && get_informed_then_do_ratio < 0.4) {
				agent->take_measure = 1;
			}
			if (agent->alert_state == 1 && get_informed_then_do_ratio < 0.8) {
				agent->take_measure = 1;
				agent->alert_state = 0;
			}
			agent->get_informed += 1;
		}
	}
	return 0;
}

__FLAME_GPU_FUNC__ int generate_warnings(xmachine_memory_flood* agent, xmachine_memory_warning_list* warning_agents) {
	
	if (TIME == 1 && agent->x == 0 && agent->y == 0) {
		float x = ((agent->x + 0.5f) / (d_message_flood_cell_width / ENV_WIDTH)) - ENV_MAX;
		float y = ((agent->y + 0.5f) / (d_message_flood_cell_width / ENV_WIDTH)) - ENV_MAX;
		float total_financial_damage = 0;
		int total_take_measure = 0;
		int total_get_warning = 0;
		int total_alert_state = 0;
		int total_sandbag1 = 0;
		int total_sandbag2 = 0;
		int total_sandbag3 = 0;
		int total_inform_others = 0;
		int total_get_informed = 0;
		int flooded_households = 0;
		add_warning_agent(warning_agents, x, y, flooded_households,
			total_financial_damage, total_take_measure, total_get_warning, total_alert_state, total_sandbag1, total_sandbag2, total_sandbag3,
			total_inform_others, total_get_informed);
	}
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
