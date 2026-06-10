#include "op_profiler.h"

#include "common.h"  // log_output
#include <iomanip>  // std::setw



std::unordered_map<std::string, OpTiming> g_op_timings;
std::mutex g_timing_mutex;

void record_op_timing(const char* opType, double elapsed_us) {
    if (!opType)
        return;
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    g_op_timings[opType].add_sample(elapsed_us);
}

void log_op_timings() {
    std::ostringstream log;
    log << "Operation timing statistics (count, avg_us, min_us, max_us):";
    if (g_op_timings.empty()) {
        log << "\n    (no operation timings collected)";
        log_output(log, true);
        return;
    }

    std::vector<std::pair<std::string, OpTiming>> sorted;
    for (const auto& [name, t] : g_op_timings)
        sorted.emplace_back(name, t);
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) {
            double avg_a = a.second.count ? (a.second.sum_us / a.second.count) : 0.0;
            double avg_b = b.second.count ? (b.second.sum_us / b.second.count) : 0.0;
            return avg_a > avg_b;
        });

    size_t max_name_len = 0;
    int max_count_len = 0, max_sum_len = 0, max_min_len = 0, max_avg_len = 0, max_max_len = 0;

    struct Formatted {
        std::string count_str, sum_str, min_str, avg_str, max_str;
    };
    std::vector<Formatted> formatted;
    formatted.reserve(sorted.size());

    for (const auto& [name, t] : sorted) {
        // имя
        size_t name_len = name.length() + 1;  // +1 для ':'
        if (name_len > max_name_len) max_name_len = name_len;

        double avg = t.count ? (t.sum_us / t.count) : 0.0;

        char buf[64];
        // count
        snprintf(buf, sizeof(buf), "%d", t.count);
        std::string c_str = buf;
        if (c_str.length() > (size_t) max_count_len) max_count_len = c_str.length();
    
        // count
        snprintf(buf, sizeof(buf), "%.3f", t.sum_us);
        std::string sum_s = buf;
        if (sum_s.length() > (size_t) max_sum_len) max_sum_len = sum_s.length();

        // min
        snprintf(buf, sizeof(buf), "%.3f", t.min_us);
        std::string min_s = buf;
        if (min_s.length() > (size_t) max_min_len) max_min_len = min_s.length();

        // avg
        snprintf(buf, sizeof(buf), "%.3f", avg);
        std::string avg_s = buf;
        if (avg_s.length() > (size_t) max_avg_len) max_avg_len = avg_s.length();

        // max
        snprintf(buf, sizeof(buf), "%.3f", t.max_us);
        std::string max_s = buf;
        if (max_s.length() > (size_t) max_max_len) max_max_len = max_s.length();

        formatted.push_back({c_str, sum_s, min_s, avg_s, max_s});
    }

    int idx = 0;
    for (const auto& [name, t] : sorted) {
        const auto& fmt = formatted[idx++];
        log << "\n    " << name << ":"
            << std::string(max_name_len - name.length() - 1, ' ')
            << " count=" << std::right << std::setw(max_count_len) << fmt.count_str
            << " | min=" << std::setw(max_min_len) << fmt.min_str << " us"
            << " | avg=" << std::setw(max_avg_len) << fmt.avg_str << " us"
            << " | max=" << std::setw(max_max_len) << fmt.max_str << " us"
            << " | sum=" << std::setw(max_sum_len) << fmt.sum_str << " us";
    }
    log_output(log, true);
}

void reset_op_timings() {
    std::lock_guard<std::mutex> lock(g_timing_mutex);
    g_op_timings.clear();
}
