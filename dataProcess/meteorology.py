# -*- coding:UTF-8 -*-
import requests
from bs4 import BeautifulSoup



def web(year, month):
    if month<10:
        website = "http://lishi.tianqi.com/shanghai/"+str(year)+"0"+str(month)+".html"
    else:
        website = "http://lishi.tianqi.com/shanghai/"+str(year)+str(month)+".html"
    return website

def main():
    year = 2014
    begin_month = 7
    end_month = 12
    for month in range(begin_month, end_month+1):
        tempweb = requests.get(web(year, month)).text
        tempbs = BeautifulSoup(tempweb, 'lxml').find('div', class_='tqtongji2').find_all('ul')[1:]
        with open('../data/meteorology/'+str(year)+str(month)+'.txt', 'w') as file:
            for oneday in tempbs:
                oneday = oneday.find_all('li')
                print(type(str(oneday[0].get_text())))
                temp = oneday[0].get_text().encode("utf-8")+','+oneday[3].get_text().encode("utf-8")\
                       +','+oneday[4].get_text().encode("utf-8")+','+oneday[5].get_text().encode("utf-8")+'\n'
                file.write(temp)

def analysis():
    year = 2014
    month = 7
    tq, direction, power = {}, {}, {}

    for month in range(7, 13):
        with open('../data/meteorology/' + str(year) + str(month) + '.txt', 'r') as file:
            for line in file:
                line = line.split(',')[1:]
                tmp_tq = line[0].split('转')[0]
                if tmp_tq in tq:
                    tq[tmp_tq] += 1
                else:
                    tq[tmp_tq] = 1
                if line[1] in direction:
                    direction[line[1]] += 1
                else:
                    direction[line[1]] = 1
                if line[2] in power:
                    power[line[2]] += 1
                else:
                    power[line[2]] = 1
    for key in direction:
        print(key)
        print(direction[key])

if __name__ == "__main__":
    analysis()
