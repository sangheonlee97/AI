def ad(c, b):
    for i, v in enumerate(c):
        c[i] = v + b
    return c

    
a = [1, 2, 3]
b = ad(a, 3)
print(a)
print(b)