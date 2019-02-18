#ifndef __STOP_WATCH__
#define __STOP_WATHC__

#include <ctime>
#include <chrono>
#include <iostream>

class StopWatch
{
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::milliseconds ms;

public:
  StopWatch(bool run = false);
  void reset();
  ms Elapsed() const;

private:
  clock::time_point start;
};

std::ostream &operator<<(std::ostream &os, const StopWatch &timer);

#endif