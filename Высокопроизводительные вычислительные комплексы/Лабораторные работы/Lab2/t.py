from math import factorial as fact
from functools import reduce

def p_0(R, N):
    num = R ** N
    den = fact(N - 1) * (N - R)
    s = sum((R ** n) / fact(n) for n in range(N))
    return 1 / (num / den + s)

  
print(p_0(6, 5))