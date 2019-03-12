
mode = 'test'
city = 'chengdu'
grid = 50
ori_file = '../data/DIDI/'+mode+'/'+mode+'-'+city+'-20.txt'
grid_file = '../data/'+mode+'/'+mode+'-'+city+'-grid'+str(grid)+'-20.txt'
select_ori_file = '../data/'+mode+'/'+mode+'-'+city+'-20-selected.txt'
ori_tra_set, grid_tra_set = set(), set()
with open(grid_file, 'r') as file:
    for line in file:
        line = line.strip().split(',')[0]
        grid_tra_set.add(line)

with open(ori_file, 'r') as file:
    for line in file:
        tra = line.strip().split('\t')[0]
        ori_tra_set.add(tra)
        if tra not in grid_tra_set:
            continue
        with open(select_ori_file, 'a') as file:
            file.write(line)
print(set(ori_tra_set).__len__(), set(grid_tra_set).__len__())
# print(set(ori_tra_set) - set(grid_tra_set))
print(set(grid_tra_set) - set(ori_tra_set))