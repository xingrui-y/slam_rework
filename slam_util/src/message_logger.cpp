#include "message_logger.h"
#include <iostream>

std::mutex MessageLogger::stdout_mutex_;

void MessageLogger::log(const char *str)
{
    std::lock_guard<std::mutex> lock(stdout_mutex_);
    std::cout << str << std::endl;
}