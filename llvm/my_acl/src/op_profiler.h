#ifndef OP_PROFILER_H
#define OP_PROFILER_H

#include <unordered_map>
#include <string>
#include <mutex>

struct OpTiming {
    int      count = 0;
    double   sum_us = 0.0;
    double   min_us = std::numeric_limits<double>::max();
    double   max_us = 0.0;

    void add_sample(double us) {
        ++count;
        sum_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
};

extern std::unordered_map<std::string, OpTiming> g_op_timings;
extern std::mutex g_timing_mutex;

void record_op_timing(const char* opType, double elapsed_us);
void log_op_timings();
void reset_op_timings();

#endif // OP_PROFILER_H
