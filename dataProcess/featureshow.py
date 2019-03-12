import json

# lonMin=121.44
# lonMax=121.48
# latMin=31.20
# latMax=31.24
lonMin = 104.00
lonMax = 104.13
latMin = 30.60
latMax = 30.73

preStr = "{ \n \"type\": \"FeatureCollection\", \n \"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\" } }, \"features\": [\n"
endStr = "] \n }"
types = {}
class Road:
    '''a road'''
    def __init__(self, roadId, type, tunnel, bridge, oneway):
        self.roadId = roadId
        self.type = type
        self.tunnel = tunnel
        self.bridge = bridge
        self.oneway = oneway
        self.vertices = []
        self.lonMax = 0
        self.lonMin = 10000
        self.latMax = 0
        self.latMin = 10000
        self.length = -1

    def inroad(self, point, v1, v2):
        x0 = point[0]
        y0 = point[1]
        x1 = v1[0]
        y1 = v1[1]
        x2 = v2[0]
        y2 = v2[1]
        if x0<=max(x1, x2) and x0>=min(x1, x2) and y0<=max(y1, y2) and y0>=min(y1, y2):
            return True
        return False
    def addVertice(self, v):
        self.vertices.append(v)
    def calBox(self):
        '''border of a road'''
        for v in self.vertices:
            if v[0]>self.lonMax:
                self.lonMax = v[0]
            if v[0]<self.lonMin:
                self.lonMin = v[0]
            if v[1]>self.latMax:
                self.latMax = v[1]
            if v[1]<self.latMin:
                self.latMin = v[1]
            #print self.latMax, self.latMin
index = 0
typesget = [u'primary', u'pedestrian', u'bridleway', u'secondary_link', u'monorail', u'tertiary',
         u'primary_link', u'service', u'residential', u'motorway_link', u'cycleway', u'preserved',
         u'secondary', u'light_rail', u'living_street', u'narrow_gauge', u'track', u'rail',
         u'motorway', u'tertiary_link', u'trunk', u'path', u'pier', u'trunk_link', u'footway',
         u'tram', u'unclassified', u'steps', u'subway', u'raceway', u'disused', u'road']
print(len(typesget))
def loadRoadNetJson(injsonFileName, index1):
    '''load road network from geojson file'''
    fway = open(injsonFileName, 'r')
    outjson = '../data/view/features/ntypes'+str(index1)+'.geojson'
    outway = open(outjson, 'w')
    roadJson = json.load(fway, encoding="utf-8")
    print 'begin to read roads'
    # gridIndex = 0
    index = 0
    outway.write(preStr)
    for road in roadJson['features']:
        roadId = road["properties"]["id"]
        roadType = road["properties"]["type"]
        # print type(roadType)
        if not types.has_key(roadType):
            types[roadType] = index
            index = index + 1
        roadTunnel = road["properties"]["tunnel"]
        roadBridge = road["properties"]["bridge"]
        roadOneway = road["properties"]["oneway"]
        roadTmp = Road(roadId, roadType, roadTunnel, roadBridge, roadOneway)
        # print(roadTunnel)
        roadTmp.vertices = road["geometry"]["coordinates"]
        roadTmp.calBox()
        if not (roadTmp.lonMin > lonMax or roadTmp.lonMax < lonMin or roadTmp.latMin > latMax or roadTmp.latMax < latMin):
            if roadType == typesget[index1]:
            # if roadOneway == index1:
                outway.write(json.dumps(road)+',\n')
                index += 1
    # print(index1, index)
    outway.write(endStr)
    # print 'types is: ', len(types), types
    print 'road file reading done...'



inpath = '../data/chengdu.geojson'

for i in range(len(typesget)):
    loadRoadNetJson(inpath, i)
# loadRoadNetJson(inpath, 0)