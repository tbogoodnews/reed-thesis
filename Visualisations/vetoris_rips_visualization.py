import matplotlib.pyplot as plt
import math
from random import uniform
import random
random.seed(1234)

# This makes a series of circles, then pprints out code for tikz

rand_radii = [uniform(0, 2 * math.pi) for j in range(25)]
noise_f = lambda d : uniform(-d/2, d/2)

n = lambda z = 0.01 : noise_f(z)

radii = [(math.pi * 2  * (i/ 10)) + n(0.4) for i in range(10)]
circle_a = [(0.8 * math.cos(i), 0.8 * math.sin(i)) for i in radii]


radii = [(math.pi * 2  * (i/ 7)) + n(0.2) for i in range(7)]
circle_b = [(0.5 * math.cos(i) + 1.6, 0.5 * math.sin(i) + 1.3) for i in radii]

radii = [(math.pi * 2  * (i/ 5)) + n(0.2) for i in range(5)]
circle_c = [((0.3 * math.cos(i)) + 1.6, (0.3 * math.sin(i)) - 0.8) for i in radii]

points = circle_a + circle_b + circle_c
x = [i[0] for i in points]
y = [i[1] for i in points]


plt.scatter(x, y)
plt.show()


for p in points:
    letter = chr(points.index(p) + 65)
    print("\\coordinate (", letter, ") at (", round(p[0], 3), ",", round(p[1], 3), ");", sep = "")


dist = 1 / 2 # This is epsilon
for p in points:
    for j in points[points.index(p) + 1:]:
        if math.sqrt((p[0] - j[0])**2 + (p[1] - j[1])**2) < dist * 2:
            print("\\draw (", chr(points.index(p) + 65), ") -- (", chr(points.index(j) + 65), ");", sep = "")
