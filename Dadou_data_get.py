'''
@Author: Fan LR
@Date: 2020-03-21 20:11:50
@e_mail: 1243124253@qq.com
@LastEditTime: 2020-04-06 20:59:36
@Descripttion: 
'''
#%%
import cv2
import numpy as np
import os
from tqdm._tqdm import trange
from keras.utils import np_utils
def data_get():
    n_class=10
    data_name_list=os.listdir('/home/flr/Desktop/Dadou/')
    data_x=[]
    y_list=[]
    index=np.arange(7218)
    np.random.shuffle(index)
    for i in trange(len(data_name_list)):
        picture_data=cv2.imread('/home/flr/Desktop/Dadou/'+data_name_list[i],1)
        data_x.append(picture_data)
        y_list.append(data_name_list[i][0].split('-')[0])
    data_y=np_utils.to_categorical(y_list, 17)
    index_train=index[0:6000]
    index_test=index[6000:7218]
    data_x_train=np.array(data_x)[index_train,:,:,:]
    data_y_train=np.array(data_y)[index_train,:]
    data_x_test=np.array(data_x)[index_test,:,:,:]
    data_y_test=np.array(data_y)[index_test,:]
    return data_x_train,data_y_train,data_x_test,data_y_test,n_class
# if __name__ == "__main__":
#     data_x_train,data_y_train,data_x_test,data_y_test,n_class=data_get()



# %%
