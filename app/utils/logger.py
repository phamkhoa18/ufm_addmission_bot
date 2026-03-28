"""
Logger Module — Hệ thống ghi nhật ký chuẩn doanh nghiệp.

Thay thế toàn bộ print() trong các Node bằng logger chuyên nghiệp.
Hỗ trợ ghi ra Console + File log (nếu cấu hình).

Cách dùng:
  from app.utils.logger import get_logger
  logger = get_logger(__name__)

  logger.info("RAG Node - Hybrid Search hoàn tất: 6 parents")
  logger.warning("Top1 cosine=0.72 < threshold 0.85")
  logger.error("Lỗi kết nối DB", exc_info=True)
"""

import logging
import os
import sys


# Định dạng log chuẩn: [THỜI GIAN] [CẤP ĐỘ] [MODULE] Nội dung
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cấp độ log mặc định (có thể override qua biến môi trường)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Thư mục lưu file log (tạo nếu chưa có)
_LOG_DIR = os.getenv("LOG_DIR", "")


def get_logger(name: str) -> logging.Logger:
    """
    Tạo hoặc lấy logger theo tên module.

    Args:
        name: Thường dùng __name__ để tự nhận diện module.

    Returns:
        logging.Logger đã được cấu hình sẵn.
    """
    logger = logging.getLogger(name)

    # Tránh gắn handler trùng nếu gọi nhiều lần
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    # Handler 1: Console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(console_handler)

    # Handler 2: File log (nếu LOG_DIR được cấu hình)
    if _LOG_DIR:
        os.makedirs(_LOG_DIR, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(_LOG_DIR, "ufm_bot.log"),
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(file_handler)

    # Ngăn log bị lan ra root logger
    logger.propagate = False

    return logger
