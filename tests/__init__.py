import mango

model = mango.AdaptiveVGGNet16

n = 2
s = (1000, 1000, 1)

model(n, s).summary()