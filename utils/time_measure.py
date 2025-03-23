import torch


def time_decorator(name, timings):
    '''
    Decorator for measuring function elapsed time.

    Parameters.
        name: str, name to use when logging (e.g. “rope_forward”)
        timings: dict, dictionary to store the results in
    '''

    def decorator(fn):
        def wrapper(*args, **kwargs):
            import torch
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            result = fn(*args, **kwargs)

            end.record()
            torch.cuda.synchronize()
            timings[name] = start.elapsed_time(end)
            return result

        return wrapper

    return decorator
def record_time(name,timings):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    def stop():
        end.record()
        torch.cuda.synchronize()
        timings[name] = start.elapsed_time(end)
    return stop