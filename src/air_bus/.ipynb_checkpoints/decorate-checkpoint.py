import cProfile, pstats, io, os, time



########  decorators that I might need


def time_func(func):
    def inner(*args, **kwargs):
        start = time.time()
        z = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print("function call took %f seconds" %elapsed)
        return z
    return inner


def profile(fnc):
    """"decorator that uses cProfile to look at a function"""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr,stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return wrapper  