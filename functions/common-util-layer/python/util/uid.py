"""Unique ID utilities for SageMaker"""

from datetime import datetime


def append_timestamp(s: str, sep: str="-") -> str:
    """Append current datetime to `s` in a format suitable for SageMaker job names"""
    return s + datetime.now().strftime(
        f"{sep}%Y{sep}%m{sep}%d{sep}%H{sep}%M{sep}%S"
    )
