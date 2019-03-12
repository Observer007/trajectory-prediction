# 程序说明

## gpsVectorGeneration.py
- 最原始的根据gps点的轨迹数据，划分成序列，作为RNN的输入，每两个点之间的时间间隔不一定相同。这个程序不用了

## gpsVectorGenerationSameInterval2.py
- 将原始上海的gps数据划分成等时间间隔的经纬度序列，线性插值

## RoadNetwork2.py
- 根据mapmatching之后生成的轨迹序列 gpsVector 映射到路网上，使用每个路段的id来表示每个轨迹点
- 输入是gpsVector
- 输出是gpsVectorGrid

## split_holiday.py
- 结果划分是否是假期或早中晚

## changeFormat.py & changeFormat1.py
- 是把经纬度序列，转变成geojson格式，可以用qgis显示出来

## coord_transform_util.py
- 坐标系转化，didi坐标系和osm坐标系不同

## average_trajectory_length.py
- 平均轨迹距离，速度

## extend.py
- 将不同步轨迹切分成相等的10步
