def wrap_output(wrapper):
    def wrap_helper(func):
        def wrapped_func(*args, **kwargs):
            return (wrapper(xx) for xx in func(*args, **kwargs))
        return wrapped_func
    return wrap_helper