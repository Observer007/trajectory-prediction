
import math


def cal_distance(point1, point2):
    return math.sqrt(((point1[0]-point2[0])*math.cos(point1[0]/180*math.pi)*111000)**2+((point1[1]-point2[1])*111000)**2)


def cal_trajectory_length(trajectory):
    distance = 0
    for i in range(1, len(trajectory)):
        distance += cal_distance(trajectory[i], trajectory[i-1])
    return distance


def convert(line):
    points = []
    for i in range(len(line)/3):
        point = [float(line[3*i]), float(line[3*i+1])]
        points.append(point)
    return points


if __name__ == '__main__':
    size = 'small'
    raw_data_path = '../data/train/train-'+size+'-20.txt'
    distance = 0
    with open(raw_data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split('\t')[1:]
            trajectory = convert(line)
            temp = cal_trajectory_length(trajectory)
            print(temp)
            distance += temp
        print(len(lines))
        distance = distance/len(lines)
    print(distance)