import numpy as np
import math, os
import time
city = 'beijing'
hidden_dim = 256        # 256 for chengdu beijing 128 for shanghai
inter_dim = 256
intra_dim = 34             # 33 chengdu 34 beijing 38 shanghai
ext_dim = 9
num_step = 9
grid = 300
save_baseline = True
save_weekday = False
predict_rate = True
print(inter_dim, intra_dim, ext_dim)
select_file = '../data/test/test-'+city+'-selectid-'+str(grid)+'.txt'
baseline_file = '../data/test/test-'+city+'-baseline.txt'
weekday_file = '../data/test/test-'+city+'-weekday.txt'
weekend_file = '../data/test/test-'+city+'-weekend.txt'
if not os.path.exists(select_file) or city == 'shanghai' or city == 'beijing':
    select = True
    selectids = []
else:
    select = False
    selectids = []      # not use to cal
    with open(select_file, 'r') as file:
        for line in file:
            line = line.strip()
            selectids.append(line)

class Trajectory:
    def __init__(self, id, grids, time=None, coors=None):
        self.id = id
        self.grids = grids
        self.coors = coors
        self.time = time
    # ================= change grid to vertice =======================
    def change_grid2coor(self, grid2cor):
        self.coors = []
        for grid in self.grids:
            self.coors.append(grid2cor[grid])
    def set_time(self, time):
        self.time = time
class Trajectorys:
    def __init__(self, trajectorys):
        self.id2tra = {}
        for tra in trajectorys:
            self.id2tra[tra.id] = tra

def DistanceBetweenMeter(geo1, geo2):
    R = 6378137
    lonA, latA = geo1[0]/180*math.pi, geo1[1]/180*math.pi
    lonB, latB = geo2[0]/180*math.pi, geo2[1]/180*math.pi
    return R*math.acos(min(1.0, math.sin(math.pi/2-latA)*math.sin(math.pi/2-latB)*
        math.cos(lonA-lonB) + math.cos(math.pi/2-latA)*math.cos(math.pi/2-latB)))
# =============== read road grid info ============================
def read_road_info(road_file):
    grid_num = 0
    with open(road_file, 'r') as file:
        for line in file:
            if line.strip().__len__()>3:
                grid_num+=1
    grid2cor = []
    with open(road_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            grid2cor.append([float(line[1]), float(line[2])])
    # print(DistanceBetweenMeter(grid2cor[2472], grid2cor[2408]))
    assert grid2cor.__len__() == grid_num
    return grid2cor
def weekday_judge(city, date):
    '''shanghai'''
    if city == 'shanghai':
        date = date // 100000
        year = '2014'
        month = (date - 1488) // 31
        day = (date - 1488) % 31
        if month < 10:
            month = '0' + str(month)
        elif month>12:
            month = '0'+str(month%12)
            year='2015'
        else:
            month = str(month)
        if day < 10:
            if day == 0:
                day = 1
            date = year + month + '0'+str(day)
        else:
            date = year + month + str(day)
        # print(date)
        vec = time.strptime(date, '%Y%m%d').tm_wday
        if vec<5:
            return 0
        else:
            return 1
    '''chengdu'''

    if city == 'chengdu':
        date = date - 1477988400
        date = int(date//86400)+1
        if date < 10:
            date = '2016110'+str(date)
        else:
            date = '201611'+str(date)
        vec = time.strptime(date, '%Y%m%d').tm_wday
        # print(vec)
        if vec < 5:
            return 0
        else:
            return 1
    if city == 'beijing':
        date = date // 86400
        year = date // 365
        month = date % 365 // 31
        day = date % 365 % 31
        str_year = str(year)
        if month < 10:
            str_month = '0' + str(month)
        else:
            str_month = str(month)
        if day < 10:
            str_day = '0' + str(day)
        else:
            str_day = str(day)
        date = str_year + str_month + str_day
        vec = time.strptime(date, '%Y%m%d').tm_wday
        tempweek = np.zeros(9)
        tempweek[vec] = 1
        if vec < 5:
            return 0
        else:
            return 1

def peak_time(city, _date):
    hour = 0
    '''shanghai'''
    if city == 'shanghai':
        date = _date % 100000
        hour = date // 3600

    '''chengdu'''

    if city == 'chengdu':
        date = _date - 1477958400
        # date = int(date//86400)+1
        date = date % 86400
        hour = date // 3600
        if hour >= 6 and hour <= 9:
            if not weekday_judge(city, _date):
                return 1
            else:
                return -1
        elif hour >= 17 and hour <= 20:
            if not weekday_judge(city, _date):
                return 2
            else:
                return -1
        else:
            if not weekday_judge(city, _date):
                return 0
            else:
                return -1
    '''beijing'''
    if city == 'beijing':
        date = _date % 86400
        hour = date//3600
        if hour >= 6 and hour <= 9:
            if not weekday_judge(city, _date):
                return 1
            else:
                return -1
        elif hour >= 17 and hour <= 20:
            if not weekday_judge(city, _date):
                return 2
            else:
                return -1
        else:
            if not weekday_judge(city, _date):
                return 0
            else:
                return -1
# ================= cal loss ======================================

def calDistance(tras1, tras2, tras_id=None, tras_time=None, start=5):
    assert tras1.__len__() == tras2.__len__()
    dis_error = np.zeros([5])
    morning_dis_error, evening_dis_error, usual_dis_error = np.zeros([5]), np.zeros([5]), np.zeros([5])
    weekday_dis_error, holiday_dis_error = np.zeros([5]), np.zeros([5])
    morning_count, evening_count, usual_count, weekday_count, holiday_count = \
        np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5]), np.zeros([5])
    num_count = np.zeros([5])
    distribution_error = np.zeros([5, 5])
    min_error, max_error = np.ones([5])*100000, np.zeros([5])
    for t in range(len(tras1)):
        tra1 = tras1[t][start:]
        tra2 = tras2[t][start:]
        flag = True
        tmp_dis_error = np.zeros_like(dis_error)
        tmp_num_count = np.zeros_like(num_count)
        tmp_max_error, tmp_min_error = np.zeros([5]), np.ones([5])*100000
        tmp_distribution_error = np.zeros_like(distribution_error)
        for i in range(5):
            dis = DistanceBetweenMeter(tra1[i], tra2[i])
            if select and dis > 5000:
                if city == 'chengdu':
                    if tras_id[t].split('_')[0] not in selectids:
                        selectids.append(tras_id[t].split('_')[0])
                    # print(tras_id[t])
                    flag = False
                    if peak_time(city, int(tras_time[t])) == 1:
                        pass
                elif city == 'shanghai':
                    if tras_id[t] not in selectids:
                        selectids.append(tras_id[t])
                    flag = False
                    if peak_time(city, int(tras_time[t])) == 1:
                        pass
                # elif city == 'beijing':
                #     if tras_id[t].split('_')[0] not in selectids:
                #         selectids.append(tras_id[t].split('_')[0])
                #     flag = False
            if predict_rate:
                flag = True
                if dis > 8000:
                    continue

            tmp_dis_error[i] += dis
            tmp_num_count[i] += 1

            if dis > 4000:
                tmp_distribution_error[i][4] += 1
            else:
                tmp_distribution_error[i][int(dis)//1000] += 1
            tmp_min_error[i] = min(tmp_min_error[i], dis)
            tmp_max_error[i] = max(tmp_max_error[i], dis)
        if flag:
            time_judge = peak_time(city, int(tras_time[t]))
            if time_judge==-1:
                holiday_dis_error += tmp_dis_error
                holiday_count += tmp_num_count
            else:
                weekday_dis_error += tmp_dis_error
                weekday_count += tmp_num_count
                if time_judge == 1:
                    # print('morning')
                    morning_count += tmp_num_count
                    morning_dis_error += tmp_dis_error
                elif time_judge == 2:
                    # print('evening')
                    evening_count += tmp_num_count
                    evening_dis_error += tmp_dis_error
                else:
                    usual_dis_error += tmp_dis_error
                    usual_count += tmp_num_count
            dis_error += tmp_dis_error
            num_count += tmp_num_count
            distribution_error += tmp_distribution_error
            for i in range(5):
                min_error[i] = min(min_error[i], tmp_min_error[i])
                max_error[i] = max(max_error[i], tmp_max_error[i])
    tra_num = num_count[0]
    dis_error = dis_error/num_count
    distribution_error = distribution_error/num_count
    weekday_dis_error = weekday_dis_error/weekday_count
    holiday_dis_error = holiday_dis_error/holiday_count
    morning_dis_error = morning_dis_error/morning_count
    evening_dis_error = evening_dis_error/evening_count
    usual_dis_error   = usual_dis_error/usual_count
    num_count = num_count/len(tras1)
    return tra_num, dis_error, num_count, distribution_error, min_error, max_error,\
            weekday_dis_error, holiday_dis_error, morning_dis_error, evening_dis_error, usual_dis_error
def read_result(result_file, grid2cor):
    index = 0
    pre_list, gt_list = [], []
    with open(result_file, 'r') as file:
        for line in file:
            if line.__len__()<5:
                continue
            line = line.strip().split(',')
            id = line[0]

            grids = [int(float(i)) for i in line[1:11]]
            if index%2:
                pre_tra = Trajectory(id, grids)
                pre_tra.change_grid2coor(grid2cor)
                pre_list.append(pre_tra)
            else:
                gt_tra = Trajectory(id, grids)
                gt_tra.change_grid2coor(grid2cor)
                gt_list.append(gt_tra)
            index+=1
    # for i in range(len(pre_list)):
    #     for j in range(pre_list[i].grids.__len__()):
    #         tmp = DistanceBetweenMeter(grid2cor[pre_list[i].grids[j]],
    #                                    grid2cor[gt_list[i].grids[j]])
    #         if tmp>200:
    #             print(tmp)
    #             print(pre_list[i].id)
    #             break

    gt_list = Trajectorys(gt_list)
    pre_list = Trajectorys(pre_list)

    return gt_list, pre_list

def read_origin(origin_file):
    origin_tras = []
    with open(origin_file, 'r') as file:
        for line in file:
            if line.__len__()<5:
                continue
            line = line.strip().split('\t')
            assert line.__len__() == 34
            id = line[0]
            cor_line = line[1:]
            coors = []
            for i in range(10):
                coors.append([float(cor_line[i*3]), float(cor_line[i*3+1])])
            tra = Trajectory(id, None, coors=coors, time=int(float(cor_line[2])))
            origin_tras.append(tra)
            for i in range(len(tra.coors)-1):
                tmp = DistanceBetweenMeter(tra.coors[i], tra.coors[i+1])
                if tmp>100:
                    # print(tra.id)
                    break
    origin_tras = Trajectorys(origin_tras)
    return origin_tras
if __name__ == '__main__':
    origin_file = '../data/test/test-'+city+'-10.txt'
    result_file = '../data/result/'+city+'/result_hidden_'+\
    str(hidden_dim) + '_inter_' + str(inter_dim) + \
    '_intra_' + str(intra_dim) +'_ext_'+\
    str(ext_dim)+'_grid_' + str(grid)+\
    '_numstep_'+str(num_step)+'.txt'
    road_file = '../data/roadnet/road-' + city + '-' + str(grid) + '.txt'


    grid2cor = read_road_info(road_file)
    origins = read_origin(origin_file)
    try:
        gt_results, pre_results = read_result(result_file, grid2cor)
    except:
        result_file = '../data/result/' + city + '/result_hidden_' + \
                      str(hidden_dim) + '_inter_' + str(inter_dim) + \
                      '_intra_' + str(intra_dim) + '_ext_' + \
                      str(ext_dim) + '_grid_' + str(grid) + '.txt'
        gt_results, pre_results = read_result(result_file, grid2cor)

    tras_ori, tras_pre, tras_id, tras_time = [], [], [], []
    for id in pre_results.id2tra:
        if id == 'zepJwecsswi2vuFB5oiB.erFeqBHaglr33_9':
            p = 1
        if id not in origins.id2tra or (not select and ((city=='chengdu' and id.split('_')[0] in selectids) or (city=='shanghai' and id in selectids))):
            continue
        gt_results.id2tra[id].set_time(origins.id2tra[id].time)
        tras_time.append(gt_results.id2tra[id].time)
        tras_ori.append(gt_results.id2tra[id].coors)
        tras_pre.append(pre_results.id2tra[id].coors)
        tras_id.append(id)

    tra_num, dis_error, num_count, distribution_error, min_error, max_error,\
        weekday_dis_error, holiday_dis_error, morning_dis_error, evening_dis_error, usual_dis_error = \
        calDistance(tras_ori, tras_pre, tras_id, tras_time)
    print('tras num is %d, average error is: %f' % (tra_num, dis_error.mean()))
    print('step error is:', dis_error)
    print('predict rate is:', num_count)
    print('distribution is:', distribution_error)
    print('min error is:', min_error)
    print('max error is:', max_error)
    print('weekday dis error is: %f' % weekday_dis_error.mean())
    print('holiday dis error is: %f' % holiday_dis_error.mean())
    print('morning dis error is: %f' % morning_dis_error.mean())
    print('evening dis error is: %f' % evening_dis_error.mean())
    print('usually dis error is: %f' % usual_dis_error.mean())

    if select:
        with open(select_file, 'w') as file:
            for id in selectids:
                file.write(id+'\n')
    if save_baseline:
        with open(baseline_file, 'w') as file:
            for id in gt_results.id2tra:
                if (city == 'chengdu' and id.split('_')[0] not in selectids) or (city=='shanghai' and id not in selectids) or (city=='beijing' and id not in selectids):
                    tmp = id
                    for i in range(gt_results.id2tra[id].coors.__len__()):
                        tmp += '\t' + str(gt_results.id2tra[id].coors[i][0]) + '\t' +\
                        str(gt_results.id2tra[id].coors[i][1]) + '\t' + str(gt_results.id2tra[id].time)
                    tmp += '\n'
                    file.write(tmp)
    if save_weekday:
        with open(weekday_file, 'w') as file:
            for id in gt_results.id2tra:
                if (city == 'chengdu' and id.split('_')[0] not in selectids) or (city=='shanghai' and id not in selectids) or (city=='beijing' and id not in selectids):
                    tmp = id
                    if not weekday_judge(city, gt_results.id2tra[id].time):
                        for i in range(gt_results.id2tra[id].coors.__len__()):
                            tmp += '\t' + str(gt_results.id2tra[id].coors[i][0]) + '\t' +\
                            str(gt_results.id2tra[id].coors[i][1]) + '\t' + str(gt_results.id2tra[id].time)
                        tmp += '\n'
                        file.write(tmp)
        with open(weekend_file, 'w') as file:
            for id in gt_results.id2tra:
                if (city == 'chengdu' and id.split('_')[0] not in selectids) or (city=='shanghai' and id not in selectids) or (city=='beijing' and id not in selectids):
                    tmp = id
                    if weekday_judge(city, gt_results.id2tra[id].time):
                        for i in range(gt_results.id2tra[id].coors.__len__()):
                            tmp += '\t' + str(gt_results.id2tra[id].coors[i][0]) + '\t' +\
                            str(gt_results.id2tra[id].coors[i][1]) + '\t' + str(gt_results.id2tra[id].time)
                        tmp += '\n'
                        file.write(tmp)