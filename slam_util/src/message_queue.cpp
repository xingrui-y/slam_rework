#include "message_queue.h"

MessageQueue::MessageQueue()
{
}

void MessageQueue::enqueue_message(int msg)
{
    std::lock_guard<std::mutex> lock(lock_);
    queue_.push(msg);
}

int MessageQueue::dequeue_message()
{
    std::lock_guard<std::mutex> lock(lock_);
    if (queue_.size() > 0)
    {
        int msg = queue_.front();
        queue_.pop();
        return msg;
    }
    else
    {
        return EMPTY_QUEUE;
    }
}
