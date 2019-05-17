#ifndef __MESSAGE_QUEUE__
#define __MESSAGE_QUEUE__

#include <mutex>
#include <queue>

class MessageQueue
{
  public:
    enum
    {
        EMPTY_QUEUE = -1,
        SYSTEM_RESTART = 0
    };

    MessageQueue();
    void enqueue_message(int msg);
    int dequeue_message();

  private:
    std::queue<int> queue_;
    std::mutex lock_;
};

#endif