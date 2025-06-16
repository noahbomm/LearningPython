import math
from math import pi
radiant = float(input("Winkel in Bogenmaß eingeben: "))

degree = radiant * 180 / math.pi
Grad = int(degree)
rest_minuten = (degree - Grad) * 60
Bogenmin = int(rest_minuten)
Bogensek = (rest_minuten - Bogenmin) * 60

print(radiant, " radiant = ", Grad, "° ", Bogenmin, "' ", Bogensek, "\"", sep = "")