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

StopWatch::ms StopWatch::Elapsed() const
{
    return std::chrono::duration_cast<ms>(clock::now() - start);
}

std::ostream &operator<<(std::ostream &os, const StopWatch &timer)
{
    os << timer.Elapsed().count() << " ms";
    return os;
}