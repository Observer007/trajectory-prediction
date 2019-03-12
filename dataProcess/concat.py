

transnum = 600
size = 'small'
grids = '40'
length = '20'
train_file = "../data/train/train-"+size+"-grid"+grids+'-'+length+".txt"
test_file = "../data/test/test-"+size+"-grid"+grids+'-'+length+".txt"
new_test_file = "../data/test/test-"+size+"-ngrid"+grids+"-"+length+".txt"

traindict = {}
with open(train_file, 'r') as f:
    for line in f:
        line = line.strip().split(',')
        traindict[line[0]] = 0

with open(test_file, 'r') as f:
    # f.write('\n')
    index = 0
    for line in f:
        line0 = line.strip().split(',')
        if not traindict.has_key(line0[0]):
            with open(train_file, 'a') as f1:
                f1.write(line)
                del(line)
                index += 1
                if index==transnum:
                    break
        else:
            with open(new_test_file, 'a') as f2:
                f2.write(line)