# coding=<utf-8>

"""
(ref) https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch06/weight_init_activation_histogram.py


* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
입력 표준화(input standardization) 테스트

"""


#%%
import numpy as np 
import matplotlib.pyplot as plt



# ================================================================= #
#                   1. Create input vectors                         #
# ================================================================= #
# %% 입력 벡터 생성; 2-tuple
np.random.seed(42)



x1 = np.random.randn(1000) * 100 + 50
x2 = np.random.randn(1000) * 10


plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')
plt.scatter(x1, x2, color='black')
plt.savefig("01_input_vector.svg")
plt.show()


input_data = np.array([x1, x2]).T
print(input_data)
print(input_data.shape)

# ================================================================= #
#                        1. Create Model                            #
# ================================================================= #
# %% 입력 벡터 생성; 2-tuple

def ReLU(x):
    return np.maximum(0, x)


node_num = 2  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장


x = input_data.T

for i in range(hidden_layer_size):
    """
    순전파 연산 
    """
    
    if i != 0:
        x = activations[i-1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 0.01

    a = np.dot(w, x)  # shape: (2, 1000)

    z = ReLU(a)

    activations[i] = z




# ================================================================= #
#                   2.Plot activated values                         #
# ================================================================= #

# %% 레이어별 활성화값 출력
"""
- 레이어별 활성화값 분포
"""
for i, a in activations.items():

    plt.title(str(i+1) + "-layer")    

    plt.axvline(x=0, color='gray')
    plt.axhline(y=0, color='gray')
    plt.scatter(a[0,:], a[1,:], color='black')
#    plt.savefig("02_input_standardization.svg")

    plt.show()
# %%
