#pragma once

#include <chrono>
#include <string>
#include <vector>



class TimeLogger {
    public:
    struct timeLog {
        int features;
        int samples;
        int n_components;
        double time_ms;
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        timeLog(int features, int samples, int n_components, std::string name, std::chrono::high_resolution_clock::time_point start_time) : features{features},
                samples{samples}, n_components{n_components}, name{name}, start_time{start_time}{
                    time_ms = -123456789.0; // To identify logs without stop()
                }
    };
    std::vector<timeLog*> logs;
    int features;
    int samples;
    int n_components;
    std::string log_name;
    TimeLogger(int features, int samples, int n_components, std::string log_name) : 
        features{features}, samples{samples}, n_components{n_components}, log_name{log_name} {}
    timeLog* start(std::string name) {
        timeLog *tl = new timeLog(this->features, this->samples, this->n_components, name, std::chrono::high_resolution_clock::now());
        logs.push_back(tl);
        return tl;
    }
    void stop(timeLog* tl) {
        auto end = std::chrono::high_resolution_clock::now();
        tl->end_time = end;
        auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - tl->start_time);
        tl->time_ms = elapsed.count() * 1e-6;
    }
};


