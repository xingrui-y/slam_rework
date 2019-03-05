#include "stop_watch.h"

StopWatch::StopWatch(bool run)
{
    if (run)
        reset();
}

void StopWatch::reset()
{
    start = clock::now();
}

StopWatch::milliseconds StopWatch::Elapsed() const
{
    return std::chrono::duration_cast<milliseconds>(clock::now() - start);
}

std::ostream &operator<<(std::ostream &os, const StopWatch &timer)
{
    os << 1000.f / timer.Elapsed().count() << " fps";
    return os;
}