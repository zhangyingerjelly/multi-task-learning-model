import os
import numpy as np
import pandas as pd
import argparse
import random
import json
import argparse
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

os.environ['CUDA_VISIBLE_DEVICES'] = "1"  

def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--epoch', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training.')
    parser.add_argument('--valid_batch_size', type=int, default=1024,
                        help='Batch size for validation.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the results (0, 1 ... any positive integer)')  
    parser.add_argument('--reg', type=int, default=1e-5,
                        help='l2 reg')  
    parser.add_argument('--embedding_dim', type=int, default=4,
                        help='Number of embedding dim.')     
    parser.add_argument('--alpha', type=float, default=0,
                        help='label constraint weight.')                 
    return parser.parse_args()

args = parse_args()

#==========================calculate auc for training/validation/testing set after one epoch=======================#
#auc mustbe calculated after one epoch, not a batch
class ROCCallback(Callback):
    def __init__(self, train_file, valid_file,test_file,batch_size,save_path):
        self.train_file=train_file
        self.valid_file=valid_file
        self.test_file=test_file
        self.batch_size=batch_size
        self.save_path=save_path

    def on_train_begin(self, logs={}):
        self.best_valid_auc_activate=0
        self.auc_train={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[],'cbc_is_activate_in_t14':[]}
        self.auc_valid={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[],'cbc_is_activate_in_t14':[]}
        self.losses_train={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[],'cbc_is_activate_in_t14':[]}
        self.losses_valid={'cbc_is_click':[],'cbc_is_apply':[],'cbc_is_credit':[],'cbc_is_activate_in_t14':[]}
        return
        
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_df = pd.read_csv(self.train_file,delimiter=',',index_col=None,chunksize=self.batch_size)
        train_preds=[[],[],[],[]]
        train_labels=[[],[],[],[]]
        for df_chunk in train_df:
            train_data,train_label=preprocess(df_chunk)
            train_prediction = self.model.predict(train_data,batch_size=self.batch_size)
            for i in range(4):
                train_preds[i].extend(train_prediction[:,i])
                if i==0:
                    name='cbc_is_click'
                elif i==1:
                    name='cbc_is_apply'
                elif i==2:
                    name='cbc_is_credit'
                else:
                    name='cbc_is_activate_in_t14'
                train_labels[i].extend(train_label[name].to_numpy())
        for i in range(4):
            if i==0:
                name="cbc_is_click"
            elif i==1:
                name='cbc_is_apply'
            elif i==2:
                name='cbc_is_credit'
            else:
                name='cbc_is_activate_in_t14'
            train_auc=roc_auc_score(train_labels[i],train_preds[i])
            self.auc_train[name].append(train_auc)
            print("  "+name+" train target:",train_auc)
        
        valid_df = pd.read_csv(self.valid_file,delimiter=',',index_col=None,chunksize=self.batch_size)
        valid_preds=[[],[],[],[]]
        valid_labels=[[],[],[],[]]
        for df_chunk in valid_df:
            valid_data,valid_label=preprocess(df_chunk)
            valid_prediction = self.model.predict(valid_data,batch_size=self.batch_size)
            
            for i in range(4):
                valid_preds[i].extend(valid_prediction[:,i])
                if i==0:
                    name='cbc_is_click'
                elif i==1:
                    name='cbc_is_apply'
                elif i==2:
                    name='cbc_is_credit'
                else:
                    name='cbc_is_activate_in_t14'
                valid_labels[i].extend(valid_label[name].to_numpy())
            
        for i in range(4):
            valid_auc=roc_auc_score(valid_labels[i],valid_preds[i])
            if i==0:
                name='cbc_is_click'
            elif i==1:
                name='cbc_is_apply'
            elif i==2:
                name='cbc_is_credit'
            else:
                name='cbc_is_activate_in_t14'
                if valid_auc>self.best_valid_auc_activate:
                    self.best_valid_auc_activate=valid_auc
                    print("save")
                    self.model.save(self.save_path)
            self.auc_valid[name].append(valid_auc)
            print("  "+name+" valid target:",valid_auc)
        
        for output_name in ('cbc_is_click','cbc_is_apply','cbc_is_credit','cbc_is_activate_in_t14'):
            self.losses_train[output_name].append(logs.get(output_name+'_loss'))
            self.losses_valid[output_name].append(logs.get('val_'+output_name+'_loss'))
        
        print("best_valid_auc:",self.best_valid_auc_activate)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return





#========================================= model structure======================================#
def norm(x): # L2-norm for embedding;
    return K.l2_normalize(x,axis=-1)

def embedding_input(name, n_in, n_out, reg): 
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1)(inp)

def int_to_float(x):
    import tensorflow as tf
    return tf.to_float(x)

def continous_input(name): 
    inp = Input(shape=(1,), dtype='int64', name=name)
    inp_2=Reshape((1, 1))(inp)
    inp_3=Lambda(int_to_float)(inp_2)
    return inp, inp_3

def model_construct(embedding_number_file,reg,embedding_dim):
    global deep_cols
    with open(embedding_number_file,'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    embedding_cols=deep_cols  
    unique_vals=load_dict
    embeddings_tensors = []
    for ec in embedding_cols:
        layer_name = ec 
        vocabulary_size=unique_vals[ec]+1
        t_inp, t_build = embedding_input(layer_name, vocabulary_size, embedding_dim, reg)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)

    # input layer
    input_layer =  [et[0] for et in embeddings_tensors]
    # embedding layer
    inp_embed = [et[1] for et in embeddings_tensors]
    # concat embedding layer
    input_concatenate_layer=concatenate(inp_embed)
    input_concatenate_layer=Lambda(lambda x:tf.reduce_sum(x,axis=1))(input_concatenate_layer)

    # tower for click
    d_click = Dense(128, activation='relu')(input_concatenate_layer)
    d_click = Dense(64, activation='relu')(d_click)
    d_click = Dense(32, activation='relu')(d_click)
    d_click_out = Dense(1)(d_click)

    # tower for apply
    d_apply = Dense(128, activation='relu')(input_concatenate_layer)
    d_apply = Dense(64, activation='relu')(d_apply)
    d_apply = Dense(32, activation='relu')(d_apply)
    d_apply= concatenate([d_click,d_apply])
    d_apply=Dense(32,activation='relu')(d_apply)
    d_apply_out=Dense(1)(d_apply)

    # tower for credit
    d_credit = Dense(128, activation='relu')(input_concatenate_layer)
    d_credit = Dense(64, activation='relu')(d_credit)
    d_credit = Dense(32, activation='relu')(d_credit)
    d_credit=concatenate([d_apply,d_credit])
    d_credit=Dense(32,activation='relu')(d_credit)
    d_credit_out = Dense(1)(d_credit)

    # tower for activate
    d_activate = Dense(128, activation='relu')(input_concatenate_layer)
    d_activate = Dense(64, activation='relu')(d_activate)
    d_activate = Dense(32, activation='relu')(d_activate)
    d_activate=concatenate([d_credit,d_activate])
    d_activate=Dense(32,activation='relu')(d_activate)
    d_activate_out = Dense(1)(d_activate)

    output_layers=[d_click_out,d_apply_out,d_credit_out,d_activate_out]
    output_layers = concatenate(output_layers,axis=-1)
    model = Model(inputs=input_layer, outputs=output_layers)
    model.summary()
    return model

#======================================loss definition=====================================#

def get_loss(y_true,y_pred):
    true_click=y_true[:,0]
    pred_click=y_pred[:,0]
    loss_click = tf.reduce_mean(binary_crossentropy(true_click,pred_click))
    true_apply=y_true[:,1]
    pred_apply=y_pred[:,1]
    loss_apply = tf.reduce_mean(binary_crossentropy(
                            true_apply,
                            pred_apply))
    true_credit=y_true[:,2]
    pred_credit=y_pred[:,2]
    loss_credit = tf.reduce_mean(binary_crossentropy(
                            true_credit,
                            pred_credit))
    true_activate=y_true[:,3]
    pred_activate=y_pred[:,3]
    loss_activate = tf.reduce_mean(binary_crossentropy(
                            true_activate,
                            pred_activate))
    loss=tf.add_n([loss_click,loss_apply,loss_credit,loss_activate])
    # label_constrain
    label_constraint_click_apply = tf.maximum(pred_apply-pred_click,tf.zeros_like(pred_click))
    label_constraint_apply_credit = tf.maximum(pred_credit-pred_apply,tf.zeros_like(pred_apply))
    label_constraint_credit_activate = tf.maximum(pred_activate-pred_credit,tf.zeros_like(pred_credit))
    loss = loss + args.alpha * tf.reduce_mean(label_constraint_click_apply,label_constraint_apply_credit,
                                                        label_constraint_credit_activate,axis=0)
    return loss

def cbc_is_click_loss(y_true,y_pred): 
    return tf.reduce_mean(binary_crossentropy(y_true[:,0],y_pred[:,0]))
def cbc_is_apply_loss(y_true,y_pred):
    return tf.reduce_mean(binary_crossentropy(y_true[:,1],y_pred[:,1]))
def cbc_is_credit_loss(y_true,y_pred):
    return tf.reduce_mean(binary_crossentropy(y_true[:,2],y_pred[:,2]))
def cbc_is_activate_in_t14_loss(y_true,y_pred):
    return tf.reduce_mean(binary_crossentropy(y_true[:,3],y_pred[:,3]))


#========================================== data process for fit_generator ================================#
def preprocess(df): 
    # from dataframe format, return data (in dictionary format) and label
    global deep_cols
    datas={}
    for c in deep_cols:
        datas[c]=df[c]
    labels=df[["cbc_is_click","cbc_is_apply","cbc_is_credit",'cbc_is_activate_in_t14']]  
    return datas,labels

def steps_needed(file,batchsize):  #calculate the number batches 
    steps = 0
    df = pd.read_csv(file, chunksize=batchsize)
    for df_chunk in df:
        steps +=1 
    return steps

def csv_generator(inputPath, batchsize): 
	# open the CSV file for reading, read one batchsize into memory at a time (generator)
    while True:
        df = pd.read_csv(inputPath,delimiter=',',index_col=None,chunksize=batchsize)
        for df_chunk in df:
            datas,labels=preprocess(df_chunk)
            yield (datas,labels)


if __name__ == "__main__":
    #==========================initialize===========================#
    SEED = 2020
    # Fix numpy seed for reproducibility
    np.random.seed(SEED)
    # Fix random seed for reproducibility
    random.seed(SEED)
    # Fix TensorFlow graph-level seed for reproducibility
    tf.set_random_seed(SEED)
    tf_session = tf.Session(graph=tf.reset_default_graph())
    K.set_session(tf_session)

    #==========================file info ===========================#
    global deep_cols
    train_file='./data/sample_train.csv'
    valid_file='./data/sample_valid.csv'
    test_file='./data/sample_test.csv' 
    save_path='esmm_v3.h5'
    #'info_v4.csv' contains the name and type of input data
    feature_name_file='./data/info_v4.csv' 
    # embedding_number_v4.json record category number for every input and it will be used in embedding layer.
    embedding_number_file='./data/embedding_number_v4.json'

    #=========================input data info===========================#
    # C:category type   D: discrete type  K:other type  
    # different type has different preprocessing 
    features=pd.read_csv(feature_name_file)  
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
    deep_cols=C_features+D_features+K_features

    #=========================construct model===========================#
    model=model_construct(embedding_number_file,reg=args.reg,embedding_dim=args.embedding_dim)
    model.compile(  
        loss=get_loss,
        optimizer=Adam(learning_rate=args.lr),
        metrics=[cbc_is_click_loss,cbc_is_apply_loss,cbc_is_credit_loss,cbc_is_activate_in_t14_loss])

    # ================================training============================#
    steps_per_epoch_train=steps_needed(train_file,args.batch_size)
    steps_per_epoch_valid=steps_needed(valid_file,args.auc_batch_size)
    history=ROCCallback(train_file, valid_file,test_file,args.auc_batch_size,save_path)
    model.fit_generator(
        generator=csv_generator(train_file,args.batch_size),
        steps_per_epoch=steps_per_epoch_train,
        validation_data=csv_generator(valid_file,args.auc_batch_size),
        validation_steps=steps_per_epoch_valid,
        shuffle=True,
        callbacks=[history],
        epochs=args.epoch,
        verbose=args.verbose,
    )