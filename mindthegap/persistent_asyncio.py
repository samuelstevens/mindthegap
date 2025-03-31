import asyncio
import threading

# Thread-local storage for loops
_thread_local = threading.local()


def run(coro):
    """Run a coroutine using a persistent event loop for the current thread."""
    # Get or create the thread's event loop
    if not hasattr(_thread_local, "loop") or _thread_local.loop.is_closed():
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)

    # Run the coroutine
    return _thread_local.loop.run_until_complete(coro)
