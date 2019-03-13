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
    os << timer.Elapsed().count() << " ms";
    return os;
}