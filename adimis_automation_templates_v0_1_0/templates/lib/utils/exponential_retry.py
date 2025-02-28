import random
import asyncio  # Use asyncio for async sleep
import time
from functools import wraps
import inspect


def exponential_retry(max_retries=3, base_delay=1, max_delay=60):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries (int): Maximum number of retries.
        base_delay (float): Initial delay in seconds.
        max_delay (float): Maximum delay between retries.
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):  # For async functions

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                attempt = 0
                while attempt < max_retries:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_retries:
                            raise e
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        delay += random.uniform(0, delay * 0.1)  # Add jitter
                        print(f"Retrying in {delay:.2f} seconds... (Attempt {attempt})")
                        await asyncio.sleep(delay)  # Corrected: use asyncio.sleep()

            return async_wrapper

        else:  # For synchronous functions

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                attempt = 0
                while attempt < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempt += 1
                        if attempt >= max_retries:
                            raise e
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        delay += random.uniform(0, delay * 0.1)  # Add jitter
                        print(f"Retrying in {delay:.2f} seconds... (Attempt {attempt})")
                        time.sleep(delay)

            return sync_wrapper

    return decorator
