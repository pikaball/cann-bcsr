#include "timer.h"

std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> Timer::startTimes;
std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> Timer::endTimes;
std::map<std::string, std::vector<double>> Timer::timings;

void Log::Write(const std::string& category, const std::string& sampleName, const std::map<std::string, std::vector<double>>& timings) {
    std::string filePath = "../output/" + category + ".txt";
    std::ofstream outFile(filePath, std::ios::app);
    if (!outFile.is_open()) {
        ERROR_LOG("Failed to open log file: %s", filePath.c_str());
        return;
    }

    outFile << "Sample: " << sampleName << std::endl;
    for (const auto& pair : timings) {
        outFile << "  " << pair.first << ":" << std::endl;
        for (size_t i = 0; i < pair.second.size(); ++i) {
            outFile << "    Run " << i + 1 << ": " << std::fixed << std::setprecision(6) << pair.second[i] << " ms" << std::endl;
        }
    }
    outFile << "----------------------------------------" << std::endl;
    outFile.close();
}
