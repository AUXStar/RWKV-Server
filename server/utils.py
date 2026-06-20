import asyncio
from typing import AsyncGenerator, Callable, Any, Optional
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER


class NullLock:
    """空锁，用于不需要同步的场景"""

    def acquire(self):
        pass

    def release(self):
        pass


def nop(*args, **kwargs): ...

def finish_callback():
    """
    修复后的finish_callback，添加幂等性保护
    即使被多次调用也不会抛出InvalidStateError
    """
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    _called = False

    def callback(*args):
        nonlocal _called
        if _called or future.done():
            return
        _called = True
        loop.call_soon_threadsafe(future.set_result, args)

    return future, callback


def stream_callback():
    """
    多消费者广播模式的 stream_callback。
    回调极轻量：只向所有订阅者的 queue 发送信号（None），消费者自行读取共享数据。
    """
    loop = asyncio.get_event_loop()
    _finished = False
    _subscribers: list[asyncio.Queue] = []

    def callback(data: Any) -> None:
        """轻量回调：推理线程调用，只发信号不复制数据。"""
        if _finished:
            return
        # 向所有订阅者发信号（None = "有新数据，请自行读取"）
        dead = []
        for q in _subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, data)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                _subscribers.remove(q)
            except ValueError:
                pass

    async def generator() -> AsyncGenerator[Any, None]:
        """创建一个订阅者，返回异步生成器。"""
        my_queue: asyncio.Queue[Any] = asyncio.Queue()
        _subscribers.append(my_queue)
        try:
            while True:
                signal = await my_queue.get()
                if signal is None:  # 结束信号
                    break
                yield signal
        finally:
            try:
                _subscribers.remove(my_queue)
            except ValueError:
                pass

    def finish(data: Any = None) -> None:
        nonlocal _finished
        if _finished:
            return
        _finished = True
        # 向所有订阅者发结束信号
        for q in list(_subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, None)
            except asyncio.QueueFull:
                pass

    return generator(), callback, finish

def tokenizer(path):
    return TRIE_TOKENIZER(path)
