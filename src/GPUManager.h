#pragma once
#include "Task.h"

class GPUManager{

	long total_bytes;
	long total_used_bytes;

	GPUManager(int _device_id): device_id(_device_id);
	//Look at the queue and return the estimated time for the estimated memory usage for the job. 
	int get_seconds_for_task(Task &task);

	private:
		//Queue for the tasks
		std::queue<Task> task_queue;
		//cuda device id
		int device_id;

};
