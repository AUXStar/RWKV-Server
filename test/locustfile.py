"""
RWKV 推理服务压力测试脚本（支持自动删除任务）

使用方法:
    # 启动 Web UI 模式（可动态调整并发）
    locust -f locustfile.py --host=http://localhost:8000

    # 无头模式（阶梯式加压示例）
    locust -f locustfile.py --host=http://localhost:8000 --headless -u 100 -r 10 --run-time 10m

    # 输出 CSV 报告
    locust -f locustfile.py --headless -u 50 -r 5 --run-time 30m --csv=perf_report
"""

import json
import time
from locust import HttpUser, task, between, events
from locust.exception import StopUser


class RWKVUser(HttpUser):
    """
    模拟 RWKV 服务用户：
    - 思考时间 1~3 秒
    - 创建流式任务 → 消费 SSE 流 → 自动删除任务
    """
    wait_time = between(1, 3)

    def on_start(self):
        """每个虚拟用户启动时初始化请求参数"""
        self.payload = {
            "prompt": "请介绍RWKV模型的架构特点和优势，用中文回答。",
            "max_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.5,
            "top_k": 20,
            "stream": True
        }
        self.current_task_id = None

    @task
    def create_stream_and_delete(self):
        """
        核心测试任务：
        1. POST /v1/tasks/create（stream=True）
        2. 解析 SSE 流，记录 token 数、首包延迟
        3. 请求结束后调用 DELETE /v1/tasks/{task_id}/delete 释放资源
        """
        start_time = time.time()
        token_count = 0
        first_token_time = None
        task_id = None

        # ----- 1. 创建任务并获取 SSE 流 -----
        with self.client.post(
            "/v1/tasks/create",
            json=self.payload,
            catch_response=True,
            stream=True,
            timeout=180  # 总超时 180 秒
        ) as resp:

            if resp.status_code != 200:
                resp.failure(f"创建任务失败: HTTP {resp.status_code}")
                return

            # ----- 2. 逐行消费 SSE 流 -----
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        # 第一条消息中会包含 task_id
                        if task_id is None and "task_id" in data:
                            task_id = data["task_id"]
                            self.current_task_id = task_id
                            # 可选：记录 prefill_time（data["prefill_time"]）

                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                    except json.JSONDecodeError:
                        # 忽略无效 JSON（理论上不会出现）
                        pass
            except Exception as e:
                resp.failure(f"读取 SSE 流异常: {str(e)}")
                return

        # ----- 3. 记录整体性能指标 -----
        total_time = time.time() - start_time
        events.request.fire(
            request_type="POST",
            name="/v1/tasks/create (SSE)",
            response_time=total_time * 1000,   # Locust 使用毫秒
            response_length=token_count,      # 生成的 token 数
            exception=None,
            context={}
        )

        # 可选：单独记录首 token 延迟（方便后续分析）
        if first_token_time is not None:
            events.request.fire(
                request_type="METRIC",
                name="TTFT (Time To First Token)",
                response_time=first_token_time * 1000,
                response_length=0,
                exception=None,
                context={}
            )

        # ----- 4. 删除任务释放资源 -----
        if task_id:
            with self.client.post(
                f"/v1/tasks/{task_id}/delete?force=false",
                catch_response=True,
                timeout=10
            ) as del_resp:
                if del_resp.status_code != 200:
                    del_resp.failure(f"删除任务 {task_id} 失败: HTTP {del_resp.status_code}")
                # 成功删除无需额外处理
        else:
            # 未获取到 task_id 说明流异常（服务端未按协议返回）
            events.request.fire(
                request_type="POST",
                name="/v1/tasks/create (SSE) - missing task_id",
                response_time=total_time * 1000,
                response_length=0,
                exception=Exception("未从 SSE 流中解析到 task_id"),
                context={}
            )

    @task(0)  # 权重设为 0，默认不执行；需要时可取消注释并调整权重
    def non_stream_create_and_delete(self):
        """
        备选：非流式任务（stream=False）
        适用于对比测试，不需要 SSE 解析
        """
        payload = self.payload.copy()
        payload["stream"] = False

        start_time = time.time()
        with self.client.post("/v1/tasks/create", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"非流式创建失败: {resp.status_code}")
                return
            data = resp.json()
            task_id = data.get("task_id")
            token_count = len(data.get("result", ""))  # 注意：此处不是真实 token 数，仅供参考

        total_time = time.time() - start_time
        events.request.fire(
            request_type="POST",
            name="/v1/tasks/create (Non-stream)",
            response_time=total_time * 1000,
            response_length=token_count,
            exception=None,
            context={}
        )

        if task_id:
            self.client.post(f"/v1/tasks/{task_id}/delete?force=false", timeout=10)