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


