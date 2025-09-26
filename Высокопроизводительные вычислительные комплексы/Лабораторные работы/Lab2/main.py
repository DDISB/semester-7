from math import factorial as fact
from functools import reduce


def p_0(R, N):
    num = R ** N
    den = fact(N - 1) * (N - R)
    s = sum((R ** n) / fact(n) for n in range(N))
    return 1 / (num / den + s)


def p_n(R, n, N):
    if n <= N:
        return p_0(R, N) * (R ** n) / fact(n)
    else:
        return p_0(R, N) * (R ** n) / (fact(N) * N ** (n - N))


def p_0_1(N, p):
    num = N ** (N - 1) * p ** N
    den = fact(N - 1) * (1 - p)
    s = sum((N ** i) * (p ** i) / fact(i) for i in range(N))
    return 1 / (num / den + s)


def l(N, p):
    num = N ** (N - 1) * p ** (N + 1) * p_0_1(N, p)
    den = fact(N - 1) * (1 - p) ** 2
    return num / den


# t1 = p_n(3.2, 4, 6)
# print(round(t1, 4))

# t2 = p_n(2.2, 9, 6)
# print(round(t2, 4))

# t3 = l(1, 0.0571)
# # print(round(t3, 7))
# print("{:.8f}".format(t4))

p1 = 0.2285714286
p2 = 0.1142851701
p3 = 0.1142840817

t4 = l(1, p1)
# print(round(t3, 7))
print("{:.8f}".format(t4))

t5 = l(2, p2)
# print(round(t3, 7))
print("{:.8f}".format(t5))

t6 = l(3, p3)
# print(round(t3, 7))
print("{:.8f}".format(t6))