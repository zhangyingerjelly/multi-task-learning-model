import os
import numpy as np
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #使用gpu训练
import argparse
import random
import json
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, concatenate, Embedding, Reshape
from keras.layers import Flatten, concatenate, Lambda, Dropout,Activation
from keras.models import Model
from keras.regularizers import l2, l1_l2
from keras import regularizers
from keras.models import load_model
from keras.callbacks import Callback
from keras.utils import plot_model
from keras.initializers import glorot_uniform 
from keras import activations
from keras.layers import *
from keras import backend as K
from keras.engine.topology import *
from keras.losses import *
from mmoe import MMoE



#=======================================每个epoch以后计算auc================================#
#auc必须计算完所有样本后一起计算，一个batch内计算后求平均不准
class ROCCallback(Callback):
    def __init__(self, train_file, valid_file,test_file,batch_size):
        self.train_file=train_file
        self.valid_file=valid_file
        self.test_file=test_file
        self.batch_size=batch_size

    def on_train_begin(self, logs={}):
        self.best_valid_auc_target3=0
        self.best_test_auc_target3=0
        self.auc_train={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[]}
        self.auc_valid={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[]}
        self.auc_test={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[]}
        self.losses_train={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[]}
        self.losses_valid={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[]}
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_df = pd.read_csv(self.train_file,delimiter=',',index_col=None,chunksize=self.batch_size)
        train_preds=[[],[],[]]
        train_labels=[[],[],[]]
        for df_chunk in train_df:
            train_data,train_label=preprocess(df_chunk)
            train_prediction = self.model.predict(train_data,batch_size=self.batch_size)
            
            for index, output_name in enumerate(self.model.output_names):
                train_pre=train_prediction[index]
                train_preds[index].extend(train_pre)
                train_labels[index].extend(train_label[index].to_numpy())
                
        for i in range(3):
            if i==0:
                name="cbc_is_click"
            elif i==1:
                name="cbc_is_apply"
            else:
                name="cbc_is_credit"
            
            train_auc=roc_auc_score(train_labels[i],train_preds[i])
            self.auc_train[name].append(train_auc)
            print("  "+name+" train target:",train_auc)
            
        
        valid_df = pd.read_csv(self.valid_file,delimiter=',',index_col=None,chunksize=self.batch_size)
        valid_preds=[[],[],[]]
        valid_labels=[[],[],[]]
        for df_chunk in valid_df:
            valid_data,valid_label=preprocess(df_chunk)
            valid_prediction = self.model.predict(valid_data,batch_size=self.batch_size)
            
            for index, output_name in enumerate(self.model.output_names):
                valid_pre=valid_prediction[index]
                valid_preds[index].extend(valid_pre)
                valid_labels[index].extend(valid_label[index].to_numpy())
                
        for i in range(3):
            valid_auc=roc_auc_score(valid_labels[i],valid_preds[i])
            if i==0:
                name="cbc_is_click"
            if i==1:
                name="cbc_is_apply"
            if i==2:
                name="cbc_is_credit"
                if valid_auc>self.best_valid_auc_target3:
                    self.best_valid_auc_target3=valid_auc
                    print("save")
                    self.model.save('mmoe_valid.h5')
            self.auc_valid[name].append(valid_auc)
            print("  "+name+" valid target:",valid_auc)
        
        
        test_df = pd.read_csv(self.test_file,delimiter=',',index_col=None,chunksize=self.batch_size)
        test_preds=[[],[],[]]
        test_labels=[[],[],[]]
        for df_chunk in test_df:
            test_data,test_label=preprocess(df_chunk)
            test_prediction = self.model.predict(test_data,batch_size=self.batch_size)
            
            for index, output_name in enumerate(self.model.output_names):
                test_pre=test_prediction[index]
                test_preds[index].extend(test_pre)
                test_labels[index].extend(test_label[index].to_numpy())
                
        for i in range(3):
            test_auc=roc_auc_score(test_labels[i],test_preds[i])
            if i==0:
                name="cbc_is_click"
            elif i==1:
                name="cbc_is_apply"
            else:
                name="cbc_is_credit"
                if test_auc>self.best_test_auc_target3:
                    self.best_test_auc_target3=test_auc
                    print("save")
                    self.model.save('mmoe_test.h5')
            
            self.auc_test[name].append(test_auc)
            print("  "+name+" test target:",test_auc)
        
        
        
        for index, output_name in enumerate(self.model.output_names):
            self.losses_train[output_name].append(logs.get(output_name+'_loss'))
            self.losses_valid[output_name].append(logs.get('val_'+output_name+'_loss'))
        
        print("best_valid_auc_3:",self.best_valid_auc_target3)
        print("best_test_auc_3:",self.best_test_auc_target3)
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return





#=========================================模型结构相关函数======================================#
def norm(x): #对embedding后的tensor做l2-norm;
    return K.l2_normalize(x,axis=-1)

def embedding_input(name, n_in, n_out, reg): #构建类别型数据的input，并后接embedding层
    inp = Input(shape=(1,), dtype='int64', name=name)
    #emb=Embedding(n_in, n_out, input_length=1)(inp)
    #emb_norm=Lambda(norm)(emb)
    #return inp, Embedding(n_in, n_out, input_length=1,embeddings_regularizer=l2(reg))(inp)
    return inp, Embedding(n_in, n_out, input_length=1)(inp)

def int_to_float(x):
    import tensorflow as tf
    return tf.to_float(x)

def continous_input(name): #构建稠密型数据的input层；但在本例子中所有连续型数据都被分桶离散化为类别型，使用embedding_input
    inp = Input(shape=(1,), dtype='int64', name=name)
    #return inp, Reshape((1, 1))(inp)
    inp_2=Reshape((1, 1))(inp)
    inp_3=Lambda(int_to_float)(inp_2)
    return inp, inp_3

def model_construct():
    #根据模型结构搭建模型#
    # embedding_number_v4.json记录了每个变量对应的类别总数，用于embedding时的输入尺寸
    global deep_cols
    with open("./data/embedding_number_v4.json",'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    
    embedding_cols=deep_cols  #做embedding的变量等于所有的deep col
    unique_vals=load_dict
    embeddings_tensors = []
    reg = 1e-5
    for ec in embedding_cols:
        layer_name = ec 
        vocabulary_size=unique_vals[ec]+1
        embedding_number=4
        t_inp, t_build = embedding_input(layer_name, vocabulary_size, embedding_number, reg)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)


    # 构建 input layer
    input_layer =  [et[0] for et in embeddings_tensors]
    # 构建embedding layer
    inp_embed = [et[1] for et in embeddings_tensors]
    # concat embedding layer
    input_concatenate_layer=concatenate(inp_embed)
    input_concatenate_layer=Lambda(lambda x:tf.reduce_sum(x,axis=1))(input_concatenate_layer)
    
    mmoe_layers = MMoE(   #input shape（？，x）
    units=64,
    num_experts=4,
    num_tasks=3
    )(input_concatenate_layer)    
    
    output_info = [(1,"cbc_is_click"),(1,"cbc_is_apply"),(1,"cbc_is_credit")]
    output_layers = []
    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers): #index 代表几个task
        #task_layer shape=(?, 4)
        tower_layer = Dense(
            units=32,
            activation='relu')(task_layer)
        
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1], # define output name
            activation='sigmoid')(tower_layer)
        output_layers.append(output_layer)
    # concat 这些带标签的节点作为输出，方便计算loss
    model = Model(inputs=input_layer, outputs=output_layers)
    model.summary()
    return model


#======================================loss 定义相关函数=====================================#


#==========================================fit generator 数据处理相关================================#
def preprocess(df): #包括input处理（多输入按照名称：数据的字典格式）和label提取
    global deep_cols
    datas={}
    for c in deep_cols:
        datas[c]=df[c]
    labels=[df["cbc_is_click"],df["cbc_is_apply"],df["cbc_is_credit"]]
    return datas,labels

def steps_needed(file,batchsize):  #计算需要多少step才能读完数据
    steps = 0
    df = pd.read_csv(file, chunksize=batchsize)
    for df_chunk in df:
        steps +=1 
    return steps

def csv_generator(inputPath, batchsize): #每次读取一个batchsize的数据进入内存，防止内存不足
	# open the CSV file for reading
    while True:
        df = pd.read_csv(inputPath,delimiter=',',index_col=None,chunksize=batchsize)
        for df_chunk in df:
            datas,labels=preprocess(df_chunk)
            yield (datas,labels)
    

if __name__ == "__main__":
    
    #==========================初始化===========================#
    SEED = 1
    # Fix numpy seed for reproducibility
    np.random.seed(SEED)
    # Fix random seed for reproducibility
    random.seed(SEED)
    # Fix TensorFlow graph-level seed for reproducibility
    tf.set_random_seed(SEED)
    tf_session = tf.Session(graph=tf.reset_default_graph())
    K.set_session(tf_session)
    #==========================初始赋值===========================#
    global deep_cols
    batch_size=512
    auc_batch_size=1024 #计算auc时的batchsize可以大一些
    train_file='./data/sample_train.csv'
    valid_file='./data/sample_valid.csv'
    test_file='./data/sample_test.csv' 
    #=========================导入input 特征信息===========================#
    #info_v4 记录了input变量名称和类型；
    #C\D\K分别代表三类不同的数据，可对应不同的输入预处理方式。但本例子中都已经转化为了类别特征，将全部通过embedding进行处理
    features=pd.read_csv("./data/info_v4.csv")  
    C_features=[]
    D_features=[]
    K_features=[]
    for index, row in features.iterrows():
        if row["Type_Index"]=="C":
            C_features.append(row["Feature_Name"])
        elif row["Type_Index"]=="D":
            D_features.append(row["Feature_Name"])
        elif row["Type_Index"]=="K":
            K_features.append(row["Feature_Name"])
    #代表input用到的变量
    deep_cols=C_features+D_features+K_features 
   
    model=model_construct()
    adam_optimizer = Adam()
    # compile model
    model.compile(  
    loss={'cbc_is_click': 'binary_crossentropy', 'cbc_is_apply': 'binary_crossentropy','cbc_is_credit': 'binary_crossentropy'},
        optimizer=adam_optimizer)

     # ================================训练============================#
    steps_per_epoch_train=steps_needed(train_file,batch_size)
    steps_per_epoch_valid=steps_needed(valid_file,batch_size)
    history=ROCCallback(train_file, valid_file,test_file,auc_batch_size)
    model.fit_generator(
        generator=csv_generator(train_file,batch_size),
        steps_per_epoch=steps_per_epoch_train,
        validation_data=csv_generator(valid_file,batch_size),
        validation_steps=steps_per_epoch_valid,
        shuffle=True,
        callbacks=[history],
        epochs=10,
        verbose=1,
    )