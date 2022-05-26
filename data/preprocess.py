from numpy import source


data = './fce/m2/fce.test.gold.bea19.m2'

src = open(data, "rb")
dst = open("./input.bea2019", "wb")

for line in src:
    if line[0] == ord('S'):
        dst.write(line[2:])
