import math
a = 1333
b = 2

print(math.sqrt(2))
for x in range(100):
    # a *= 10
    # b *= 100
    e = (b - a * a) * 100
    d = e / 2 / a / 10
    a = a * 10 + int(d)
    b = b * 100
    print(a)

