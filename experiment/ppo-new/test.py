from functools import partial

def p(x, y):
    print(f'{x}, {y}')

if __name__ == '__main__':
    pfunc = partial(p, x=1, y=2)
    pfunc()