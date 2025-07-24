import os
import time
from functools import wraps


def print_processing_file(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        filename = os.path.basename(func.__code__.co_filename)
        print(f"processing file {filename}")

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"    example completed in {elapsed:.3f} seconds")
        return result

    return wrapper