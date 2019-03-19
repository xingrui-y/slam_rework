#ifndef __STOP_WATCH__
#define __STOP_WATCH__

#include <ctime>
#include <chrono>
#include <iostream>

class StopWatch
{
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::microseconds milliseconds;

public:
  StopWatch(bool run = false);
  void reset();
  milliseconds Elapsed() const;

private:
  clock::time_point start;
};

std::ostream &operator<<(std::ostream &os, const StopWatch &timer);

#endif