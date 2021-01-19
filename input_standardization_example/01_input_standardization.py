# coding=<utf-8>

"""
(ref) https://youtu.be/58fuWVu5DVU?t=336


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""


#%%
import numpy as np 
import matplotlib.pyplot as plt





# ================================================================= #
#                   1. Create input vectors                         #
# ================================================================= #
# %% 입력 벡터 생성; 2-tuple
np.random.seed(42)



x1 = np.random.randn(10) * 100
x2 = np.random.randn(10) * 10


plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')
plt.scatter(x1, x2, color='black')
plt.savefig("01_input_vector.svg")
plt.show()


# ================================================================= #
#                   2.Input standardization                         #
# ================================================================= #

# %% 표준화된 입력 벡터 
"""
- 입력 벡터가 zero-centered 된다. = 데이터의 분포가 평균 0 주변으로 흩어져 있다. 
- 데이터 분포의 분산(variance) = 1
"""


norm_x1 = (x1 - np.mean(x1)) / np.std(x1)
norm_x2 = (x2 - np.mean(x2)) / np.std(x2)


plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')
plt.scatter(norm_x1, norm_x2, color='black')
plt.savefig("02_input_standardization.svg")
plt.show()
# %%
