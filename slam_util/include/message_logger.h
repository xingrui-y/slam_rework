#ifndef __MESSAGE_LOGGER__
#define __MESSAGE_LOGGER__

#include <mutex>

class MessageLogger
{
  public:
    static void log(const char *str);

  private:
    static std::mutex stdout_mutex_;
};

#endif