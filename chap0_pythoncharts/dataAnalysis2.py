import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df = pd.read_csv('wage_conv_store.csv')

df.hourly_wage.hist(bins=10)
plt.show()

df.boxplot(column='hourly_wage', return_type='dict')
plt.show()

df.boxplot(column='hourly_wage', by = 'area1', vert=False)
plt.show()

df.boxplot(column='hourly_wage', by = 'name', vert=False)
plt.show()

data=[]

for i in range(len(df)):
    data.append([])
    for j in range(len(df)):
        data[i].append([])
        data[i].append(0)

        data[i][0]=df.name[i]
        data[i][1]=df.hourly_wage[i]

def sum(keyword):
    sum=0

    for i in range(len(df)):
        if data[i][0] == keyword:
            sum+=data[i][1]

    return sum

def count(keyword):
    count =0

    for i in range(len(df)):
        if data[i][0] == keyword:
            count += 1

    return count

def mean(sum, count):
    return sum/count

mean_total = np.mean(df.hourly_wage)
mean_0711 = mean(sum("07-11"),count("07-11"))
mean_CU = mean(sum("CU"),count("CU"))
mean_gs25 = mean(sum("gs25"),count("gs25"))

y = [mean_total,mean_0711,mean_CU,mean_gs25]
x = ['전체평균','07-11','CU','gs25']

plt.bar(x,y)
plt.ylim(5500,6000)
plt.show()

label=["07-11",'CU','gs25']
count=[count("07-11"),count("CU"),count("gs25")]
plt.figure()
plt.pie(count, labels=label)
plt.show()

print(df.name)