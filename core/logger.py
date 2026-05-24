import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """日志级别枚举"""
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    STAT = "STAT"


class Logger:
    """PHM2010项目日志记录器，支持控制台和文件双输出"""

    _instances = {}

    def __init__(
        self,
        run_id: str,
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True
    ):
        """初始化日志记录器

        Args:
            run_id: 运行唯一标识符，用于日志文件隔离
            log_dir: 日志文件存储目录
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.run_id = run_id
        self.log_dir = log_dir
        self.console_output = console_output
        self.file_output = file_output
        self._file_handler = None

        if self.file_output:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"{run_id}.log")
            self._file_handler = open(log_file, 'w', encoding='utf-8')

    def __new__(cls, run_id: str, **kwargs):
        """单例模式：相同run_id返回同一实例"""
        if run_id not in cls._instances:
            cls._instances[run_id] = super().__new__(cls)
        return cls._instances[run_id]

    def _format_message(self, level: LogLevel, message: str, cut_id: Optional[str] = None) -> str:
        """格式化日志消息

        Args:
            level: 日志级别
            message: 日志消息
            cut_id: 工件ID（可选）

        Returns:
            格式化后的日志字符串
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cut_id_str = f"{cut_id} - " if cut_id else ""
        return f"{timestamp} [{level.value}] {cut_id_str}{message}"

    def _write(self, formatted_message: str):
        """写入日志到控制台和文件

        Args:
            formatted_message: 格式化后的日志消息
        """
        if self.console_output:
            print(formatted_message)

        if self.file_output and self._file_handler:
            self._file_handler.write(formatted_message + "\n")
            self._file_handler.flush()

    def info(self, message: str, cut_id: Optional[str] = None):
        """普通流程日志

        Args:
            message: 日志消息
            cut_id: 工件ID（可选）
        """
        self._write(self._format_message(LogLevel.INFO, message, cut_id))

    def warn(self, message: str, cut_id: Optional[str] = None):
        """轻微异常日志

        Args:
            message: 日志消息
            cut_id: 工件ID（可选）
        """
        self._write(self._format_message(LogLevel.WARN, message, cut_id))

    def error(self, message: str, cut_id: Optional[str] = None, exc: Optional[Exception] = None):
        """严重错误日志

        Args:
            message: 日志消息
            cut_id: 工件ID（可选）
            exc: 异常对象（可选）
        """
        if exc:
            message = f"{message}: {type(exc).__name__}: {exc}"
        self._write(self._format_message(LogLevel.ERROR, message, cut_id))

    def stat(self, message: str):
        """统计摘要日志

        Args:
            message: 统计信息消息
        """
        self._write(self._format_message(LogLevel.STAT, message))

    @classmethod
    def generate_run_id(cls, run_seq: int = 1) -> str:
        """生成运行ID

        Args:
            run_seq: 运行序号

        Returns:
            格式化的运行ID，如 run_001_20250524_1530
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M")
        return f"run_{run_seq:03d}_{date_str}_{time_str}"

    def close(self):
        """关闭日志文件句柄"""
        if self._file_handler:
            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """析构时确保文件句柄关闭"""
        self.close()


if __name__ == "__main__":
    run_id = Logger.generate_run_id(1)
    logger = Logger(run_id)

    logger.info("Signal loaded successfully", "c1_cut12")
    logger.warn("Missing metadata for cut", "c1_cut15")
    logger.error("File not found: c_1_015.csv", "c1_cut15", FileNotFoundError("c_1_015.csv"))
    logger.stat("Total loaded: 315, Success: 300, Failed: 15")

    logger.close()
