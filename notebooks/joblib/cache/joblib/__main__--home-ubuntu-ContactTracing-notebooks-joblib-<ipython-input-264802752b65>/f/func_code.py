# first line: 1
@memory.cache
def f(x):
    print(f'Running f({x})')
    return x
