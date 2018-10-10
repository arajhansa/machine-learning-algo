import random

rangeX = (0, 50)
rangeY = (0, 50)
qty = 100  

randPoints = []
i = 0
while i<qty:
    x = random.randrange(*rangeX)
    y = random.randrange(*rangeY)
    randPoints.append((x,y))
    i += 1
    
print(randPoints)