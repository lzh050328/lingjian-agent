"""
调度训练消息消费者

负责监听RabbitMQ队列，处理CREATE和COMPLETE消息
"""

import threading
import time
from typing import Optional

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from lingjian_config import config_manager, logger
from lingjian_agent.dispatch.transport_task.service import dispatch_train_service


class DispatchTrainConsumer:
    """
    调度训练消息消费者

    监听RabbitMQ队列，处理调度训练相关的消息
    """

    def __init__(self):
        """
        初始化消费者

        Args:
            service (Optional[DispatchTrainService]): 业务服务实例
        """
        self.service = dispatch_train_service
        self.connection = None
        self.channel = None
        self._is_running = False
        self._consumer_thread = None

        # 从配置获取RabbitMQ设置
        self.rabbitmq_host = config_manager.get("rabbitmq.host")
        self.rabbitmq_port = config_manager.get("rabbitmq.port")
        self.rabbitmq_username = config_manager.get("rabbitmq.username")
        self.rabbitmq_password = config_manager.get("rabbitmq.password")
        self.virtual_host = config_manager.get("rabbitmq.vhost")

        self.exchange_name = config_manager.get("rabbitmq.aigc.exchange")
        self.queue_name = config_manager.get("rabbitmq.aigc.queue.dispatchTrain")
        self.create_routing_key = config_manager.get("rabbitmq.aigc.routingKey.create")
        self.complete_routing_key = config_manager.get("rabbitmq.aigc.routingKey.complete")

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """
        验证RabbitMQ配置
        """
        required_configs = [
            ("rabbitmq.host", self.rabbitmq_host),
            ("rabbitmq.port", self.rabbitmq_port),
            ("rabbitmq.aigc.exchange", self.exchange_name),
            ("rabbitmq.aigc.queue.dispatchTrain", self.queue_name),
            ("rabbitmq.aigc.routingKey.create", self.create_routing_key),
            ("rabbitmq.aigc.routingKey.complete", self.complete_routing_key)
        ]

        for config_name, config_value in required_configs:
            if not config_value:
                raise ValueError(f"RabbitMQ配置缺失: {config_name}")

    def _connect(self) -> bool:
        """
        连接到RabbitMQ服务器

        Returns:
            bool: 连接成功返回True，失败返回False
        """
        try:
            # 创建连接参数
            credentials = pika.PlainCredentials(
                self.rabbitmq_username, self.rabbitmq_password
            )
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=int(self.rabbitmq_port),
                virtual_host=self.virtual_host,
                credentials=credentials,
                # 设置心跳和连接超时
                heartbeat=600,
                blocked_connection_timeout=300
            )

            # 建立连接
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # 声明交换机
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type='topic',
                durable=True
            )

            # 声明队列
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True
            )

            # 绑定队列到交换机
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=self.create_routing_key
            )
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=self.complete_routing_key
            )

            logger.info(f"成功连接到RabbitMQ: {self.rabbitmq_host}:{self.rabbitmq_port}")
            return True

        except Exception as e:
            logger.error(f"连接RabbitMQ失败: {e}")
            return False

    def _on_message(self, channel, method, properties, body):
        """
        消息处理回调函数

        Args:
            channel: 通道对象
            method: 方法帧
            properties: 属性
            body: 消息体
        """
        try:
            routing_key = method.routing_key
            message_body = body.decode('utf-8')

            logger.info(f"收到消息: routing_key={routing_key}, body_length={len(message_body)}")

            # 处理消息
            success = self.service.process_message(routing_key, message_body)

            if success:
                # 消息处理成功，确认消息
                channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(f"消息处理成功并已确认: {routing_key}")
            else:
                # 消息处理失败，拒绝消息并不重新入队
                channel.basic_reject(
                    delivery_tag=method.delivery_tag,
                    requeue=False
                )
                logger.error(f"消息处理失败，已拒绝: {routing_key}")

        except Exception as e:
            logger.error(f"处理消息时发生异常: {e}")
            # 发生异常时拒绝消息
            try:
                channel.basic_reject(
                    delivery_tag=method.delivery_tag,
                    requeue=False
                )
            except:
                pass

    def _consume_messages(self):
        """
        消费消息的主循环
        """
        while self._is_running:
            try:
                if not self.connection or self.connection.is_closed:
                    logger.info("尝试连接到RabbitMQ...")
                    if not self._connect():
                        time.sleep(5)  # 连接失败后等待5秒重试
                        continue

                # 设置QoS，限制每次处理的消息数量
                self.channel.basic_qos(prefetch_count=1)

                # 开始消费消息
                self.channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self._on_message
                )

                logger.info(f"开始消费队列: {self.queue_name}")

                # 开始消费循环
                self.channel.start_consuming()

            except AMQPConnectionError as e:
                logger.error(f"RabbitMQ连接错误: {e}")
                self._reconnect()
            except AMQPChannelError as e:
                logger.error(f"RabbitMQ通道错误: {e}")
                self._reconnect()
            except Exception as e:
                logger.error(f"消费消息时发生未知错误: {e}")
                self._reconnect()

    def _reconnect(self):
        """
        重新连接RabbitMQ
        """
        try:
            if self.connection:
                self.connection.close()
        except:
            pass

        self.connection = None
        self.channel = None

        if self._is_running:
            logger.info("等待5秒后重新连接...")
            time.sleep(5)

    def start(self):
        """
        启动消费者
        """
        if self._is_running:
            logger.warning("消费者已经在运行中")
            return

        self._is_running = True
        self._consumer_thread = threading.Thread(target=self._consume_messages, daemon=True)
        self._consumer_thread.start()

        logger.info("调度训练消费者已启动")

    def stop(self):
        """
        停止消费者
        """
        if not self._is_running:
            logger.warning("消费者已经停止")
            return

        self._is_running = False

        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
        except:
            pass

        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
        except:
            pass

        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)

        logger.info("调度训练消费者已停止")

    def is_running(self) -> bool:
        """
        检查消费者是否正在运行

        Returns:
            bool: 正在运行返回True，否则返回False
        """
        return self._is_running

# 创建全局消息消费者对象
dispatch_train_consumer = DispatchTrainConsumer()