import math;

x = 1.5

print(math.sqrt(2))
for n in range(15):
    y = x * 10 ** (n + 1)
    print(int(y))
    x = (x + 2 / x) / 2
