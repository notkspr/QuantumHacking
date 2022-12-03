import random

A = random.uniform(0.1, 10000.0)
p = random.uniform(0.1, 100.0)
q = random.uniform(0.1, 100.0)
r = random.uniform(0.1, 100.0)
s = random.uniform(0.1, 100.0)

print(A,p,q,r,s)


def loss():
    # function to calculate loss
    return A-(p*q*r*s)

k = 1/100000


while abs(loss())> 0.01:
    p = max(p + loss()*k, k)
    q = max(q + loss()*k, k)
    r = max(r + loss()*k, k)
    s = max(s + loss()*k, k)
    print(loss())

print(p,q,r,s,A)
print(loss())
    