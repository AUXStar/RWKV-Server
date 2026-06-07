import asyncio
from typing import AsyncGenerator, Callable, Any, Optional


class NullLock:
    """空锁，用于不需要同步的场景"""

    def acquire(self):
        pass

    def release(self):
        pass


def nop(*args, **kwargs): ...


def stream_callback():
    """
    返回一个异步生成器和一个回调函数。

    用法:
        stream_gen, collect_cb, finish_cb = stream_callback()
        # 将 collect_cb 传入 Task 的 collect_callback 参数
        task = scheduler.new_task(..., collect_callback=collect_cb, finish_callback=finish_cb)

        # 在协程中消费流式数据
        async for chunk in stream_gen:
            print(f"Received: {chunk}")
            # 可以逐块返回给客户端
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def callback(data: Any) -> None:
        """线程安全的回调，用于接收数据块"""
        loop.call_soon_threadsafe(queue.put_nowait, data)

    async def generator() -> AsyncGenerator[Any, None]:
        """异步生成器，不断 yield 接收到的数据块，直到收到 None"""
        while True:
            chunk = await queue.get()
            if chunk is None:  # 结束信号
                break
            yield chunk

    def finish(data: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, None)
        loop.call_soon_threadsafe(queue.put_nowait, None)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    return generator(), callback, finish


def finish_callback():
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def callback(*args):
        loop.call_soon_threadsafe(future.set_result, args)

    return future, callback
