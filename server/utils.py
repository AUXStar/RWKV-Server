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
    修复后的stream_callback，添加幂等性保护
    移除了多余的None，防止队列溢出
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    _finished = False

    def callback(data: Any) -> None:
        if _finished:
            return
        loop.call_soon_threadsafe(queue.put_nowait, data)

    async def generator() -> AsyncGenerator[Any, None]:
        while True:
            chunk = await queue.get()
            if chunk is None:  # 结束信号
                break
            yield chunk

    def finish(data: Any = None) -> None:
        nonlocal _finished
        if _finished:
            return
        _finished = True
        loop.call_soon_threadsafe(queue.put_nowait, None)

    return generator(), callback, finish

def tokenizer(path):
    return TRIE_TOKENIZER(path)
