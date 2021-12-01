#pragma once

struct Task {

	long required_bytes;
	int estimated_time;
	int task_owner_id;

	Task(long _required_bytes, int _estimated_time, int _task_owner_id) : required_bytes(_required_bytes), estimated_time(_estimated_time), task_owner_id(_task_owner_id){}
};
