size = 'small'

if size == 'big':
    lonMin=121.44
    lonMax=121.48
    latMin=31.20
    latMax=31.24
elif size == 'small':
    lonMin=121.40
    lonMax=121.50
    latMin=30.92
    latMax=30.98
elif size == 'mid':
    lonMin=121.70
    lonMax=121.80
    latMin=31.00
    latMax=31.06

fileName = '../data/test/test-'+size+'-road.txt'
fileName = '../data/train/train-'+size+'-road.txt'