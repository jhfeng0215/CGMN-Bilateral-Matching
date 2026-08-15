import os
from functools import wraps

def with_env_var(var_name, value):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_value = os.environ.get(var_name)
            os.environ[var_name] = value
            try:
                return func(*args, **kwargs)
            finally:
                if original_value is None:
                    del os.environ[var_name]
                else:
                    os.environ[var_name] = original_value
        return wrapper
    return decorator

