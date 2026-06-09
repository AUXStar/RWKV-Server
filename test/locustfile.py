"""
RWKV 推理服务压力测试脚本（使用 /v1/tasks/tmp 临时任务接口）

使用方法:
    # 启动 Web UI 模式（可动态调整并发）
    locust -f test/locustfile.py --host=http://localhost:8000

    # 无头模式（阶梯式加压示例）
    locust -f test/locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 --run-time 10m

    # 输出 CSV 报告
    locust -f test/locustfile.py --headless -u 50 -r 5 --run-time 30m --csv=perf_report
"""

import json
import time

from locust import HttpUser, task, between


class RWKVUser(HttpUser):
    """
    模拟 RWKV 服务用户：
    - 思考时间 1~3 秒
    - 通过 /v1/tasks/tmp 提交任务（临时，自动清理，无需手动 delete）
    - 默认跑流式，可设权重切换非流式
    """

    wait_time = between(1, 3)

    def on_start(self):
        self.payload = {
            "prompt": "User: 请介绍RWKV模型的架构特点和优势，用中文回答。Assistant: <think></think",
            "max_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.5,
            "top_k": 20,
            "stream": True,
        }

    @task
    def tmp_stream(self):
        """
        流式临时任务：
        1. POST /v1/tasks/tmp（stream=True）
        2. 消费 SSE 流，记录首包延迟和总耗时
        """
        start_time = time.time()
        first_token_time = None

        with self.client.post(
            "/v1/tasks/tmp",
            json=self.payload,
            catch_response=True,
            name="/v1/tasks/tmp (SSE)",
            stream=True,
            timeout=180,
        ) as resp:

            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        json.loads(data_str)
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                resp.failure(f"SSE 异常: {e}")
                return

        total_time = time.time() - start_time
        resp.request_meta["response_time"] = total_time * 1000
        resp.success()

        if first_token_time is not None:
            self.environment.events.request.fire(
                request_type="METRIC",
                name="TTFT (首包延迟)",
                response_time=first_token_time * 1000,
                response_length=0,
                exception=None,
                context={},
            )

    @task(0)
    def tmp_blocking(self):
        """
        非流式临时任务：
        /v1/tasks/tmp（stream=False）立即返回 task_id，需轮询 get_result。
        采用指数退避轮询，平衡响应速度和服务器压力
        """
        start_time = time.time()

        payload = {**self.payload, "stream": False}
        print(payload)
        with self.client.post(
            "/v1/tasks/tmp",
            json=payload,
            catch_response=True,
            name="/v1/tasks/tmp (Create)",
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            data = resp.json()
            task_id = data.get("task_id")
            if not task_id:
                resp.failure("未返回 task_id")
                return

        poll_start = time.time()
        poll_interval = 1
        max_interval = 5.0
        poll_count = 0

        while True:
            poll_count += 1
            with self.client.get(
                f"/v1/tasks/{task_id}/get_result",
                catch_response=True,
                name="/v1/tasks/{task_id}/get_result",
                timeout=10,
            ) as poll_resp:
                if poll_resp.status_code != 200:
                    poll_resp.failure(f"轮询 HTTP {poll_resp.status_code}")
                    return
                result = poll_resp.json()
                if result.get("finished"):
                    break
                if time.time() - poll_start > 600:
                    poll_resp.failure("轮询超时")
                    return
            
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_interval)

        total_time = time.time() - start_time
        resp.request_meta["response_time"] = total_time * 1000
        resp.success()

        self.environment.events.request.fire(
            request_type="METRIC",
            name="非流式任务总耗时",
            response_time=total_time * 1000,
            response_length=0,
            exception=None,
            context={},
        )
        self.environment.events.request.fire(
            request_type="METRIC",
            name="平均轮询次数",
            response_time=poll_count,
            response_length=0,
            exception=None,
            context={},
        )