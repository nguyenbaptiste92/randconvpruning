import os
import requests, zipfile, io
import collections
from functools import reduce
from operator import and_,or_,concat,add
import glob
import pandas as pd
import time
from scipy import interpolate
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .utils import dataset_func_chain

zip_file_url = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"
label_dict ={"lying":1,"sitting":2,"standing":3,"walking":4,"running":5,"climbingdown":7,"climbingup":8,"jumping":9}

###############################################################################################################################################
#Function to download RealWorld HAR dataset from "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"
#The following processing is realized: Sampling sample with a windows of 3 seconds, filter incorrect values (NaN), assemble the dataset by position, normalize
###############################################################################################################################################


"""
Arguments : -data_dir (string): directory path where the data will be saved
            -positional (bool): Save the dataset as multiple dataset by position or save it in one dataset
"""
def download_realworld(data_dir,positional=False):
       
    path = os.path.join(data_dir, "RealWorld_Dataset")
    
    #Check if pre-process already done
    if positional:
        pos_path = os.path.join(path, "Positionale")
        folder = glob.glob(pos_path+os.sep+"*")
        count = 0
        dico_dataset = {}
        for fold in folder:
            if os.path.exists(os.path.join(fold, "input.npy")) and os.path.exists(os.path.join(fold, "label.npy")):
                name = os.path.basename(os.path.normpath(fold))
                dico_dataset[name]={'label' : np.load(os.path.join(fold, "label.npy")) , 'input' : np.load(os.path.join(fold, "input.npy"))}
                count +=1
        if count==7:
            print("RealWorld Positional Dataset already preprocessed")
            return dico_dataset
                
    else:
        glob_path = os.path.join(path, "Globale")
        if os.path.exists(os.path.join(glob_path, "input.npy")) and os.path.exists(os.path.join(glob_path, "label.npy")):
            print("RealWorld Globale Dataset already preprocessed")
            return {'global': {'label' : np.load(os.path.join(glob_path, "label.npy")) , 'input' : np.load(os.path.join(glob_path, "input.npy"))}}
    
    #Download from url
    if os.path.exists(path):
        print("RealWorld dataset already downloaded.")
    else:
        print("Download RealWorld dataset.")
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)
        
    if len(glob.glob(os.path.join(path, "csv_data")+os.sep+"*"))==15:
        print("RealWorld zipfile already unzip.")
    else:
        print("Unzip RealWorld zipfile.")  
        for filepath in glob.glob(path+os.sep+"*"):
            user_name = os.path.basename(os.path.normpath(filepath))
            extract_path = os.path.join(path, "csv_data",user_name)
            for zippath in glob.glob(os.path.join(filepath, "data")+os.sep+"acc_*_csv.zip"):
                action_name = os.path.basename(os.path.normpath(zippath)).split('_')[1]
                with zipfile.ZipFile(zippath, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(extract_path,action_name))
                    
            for action_path in glob.glob(extract_path+os.sep+"*"):
                for zippath in glob.glob(action_path+os.sep+"*.zip"):
                    with zipfile.ZipFile(zippath, 'r') as zip_ref:
                        zip_ref.extractall(action_path)
    
    dico_location = {}
    #Traitement
    for folder_path in glob.glob(os.path.join(path, "csv_data")+os.sep+"*"):
    
        for action_path in glob.glob(folder_path+os.sep+"*"):  
            action_name = os.path.basename(os.path.normpath(action_path))
            
            for file_path in glob.glob(action_path+os.sep+"*.csv"):
                body_name = os.path.basename(os.path.normpath(file_path)).split('_')[-1].split('.')[0]
                      
                df= pd.read_csv(file_path)
                
                #Sample echantillons of 3 seconds (3*50 car la frequence est 50 Hz.)
                print("Sample dataset")
                df_list = [df.iloc[n:n+150, :] for n in range(0, len(df), 150)]
                assert reduce(add, [len(elem) for elem in df_list])==len(df),"Split not done correctly."
                df_list = [elem for elem in df_list if len(elem)==150]
                print(len(df_list))
                
                #filter nan
                print("Filter NaN.")
                df_list = [elem for elem in df_list if not elem.isnull().values.any()]
                print(len(df_list))
                
                #Resample to 50 Hz is already done (dataset sample at 50 Hz)
                
                #Assemble into a a dataset
                input_array = np.stack([elem.iloc[:,[2,3,4]].to_numpy() for elem in df_list])
                label_array = np.array([label_dict[action_name] for elem in df_list])
                print(input_array.shape)
                print(label_array.shape)
                
                #Add to position
                if body_name not in dico_location.keys():
                    dico_location[body_name] = {'label' : [] , 'input' : []}
                dico_location[body_name]['label'].append(label_array)
                dico_location[body_name]['input'].append(input_array)
                
    #Concatenate and linear transform to put data between -1 and 1    
    print("Save processed dataset")
    final_dataset={}        
    for key,dico in dico_location.items():
        final_dataset[key] = {}
        final_dataset[key]['label'] = np.concatenate(dico['label'])
        final_dataset[key]['input'] = np.concatenate(dico['input'])
        max_input,min_input = np.max(final_dataset[key]['input']),np.min(final_dataset[key]['input'])
        final_dataset[key]['input']= 2*((final_dataset[key]['input']-min_input)/(max_input-min_input))-1
        print(final_dataset[key]['label'].shape,final_dataset[key]['input'].shape)
     
    #Save    
    if positional:
        for key,dico in final_dataset.items():
            save_path = os.path.join(path, "Positionale",key)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            np.save(save_path+os.sep+"label.npy", final_dataset[key]['label'])
            np.save(save_path+os.sep+"input.npy", final_dataset[key]['input'])
        return final_dataset
    else:
        liste_label=[]
        liste_input=[]
        for key,dico in final_dataset.items():
            liste_label.append(final_dataset[key]['label'])
            liste_input.append(final_dataset[key]['input'])
        label_array = np.concatenate(liste_label)
        input_array = np.concatenate(liste_input)
        save_path = os.path.join(path, "Globale")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        np.save(save_path+os.sep+"label.npy", label_array)
        np.save(save_path+os.sep+"input.npy", input_array)
        return {'global': {'label' : label_array , 'input' : input_array}}
        
###############################################################################################################################################
#Function to load the pre-process RealWorld HAR dataset as tf.data.Dataset
###############################################################################################################################################

"""
Arguments : -path : path of the RealWorld dataset
            -proportion (float between 0 and 1): proportion for the train/test split
            -train (bool): take the train dataset
            -inter (bool): True, use other dataset for the domain generalization settings; False, use the other body position of RealWorld for the domain generalization settings
            -source_localisation (string): if inter==False, select the body localisation for the source domain (chest,forearm,head,shin,thigh,upperarm,waist)
            -transform (liste of function): apply transformations to the dataset
"""
        
def get_realworld(path,proportion=0.8,train=True,inter=True,source_localisation="chest",transform=[]):

    transform_func=dataset_func_chain(transform)
    if inter:
        dataset=download_realworld(path,positional=False)
        x = dataset["global"]["input"]
        y = dataset["global"]["label"]
        x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=proportion,random_state=1)
        if train:
            dataset = tf.data.Dataset.from_tensor_slices({'image':x_train,'label':y_train})
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(x_train)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices({'image':x_test,'label':y_test})
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(x_test)))
        dataset = dataset.map(transform_func)
        return dataset
        
    else:
    
        def relabel(y):
            inter_dict={1:0,2:1,3:2,4:3,5:4,7:5,8:6,9:7}
            b = np.copy(y)
            for old, new in inter_dict.items():
                b[y == old] = new
            return b
            
        dataset=download_realworld(path,positional=True)
        x = dataset[source_localisation]["input"]
        y = dataset[source_localisation]["label"]
        y= relabel(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=proportion,random_state=1)
        if train:
            dt = tf.data.Dataset.from_tensor_slices({'image':x_train,'label':y_train})
            dt = dt.apply(tf.data.experimental.assert_cardinality(len(x_train)))
            dt = dt.map(transform_func)
            return dt
        else:
            dict_dataset = {}
            dt = tf.data.Dataset.from_tensor_slices({'image':x_test,'label':y_test})
            dt = dt.apply(tf.data.experimental.assert_cardinality(len(x_test)))
            dt = dt.map(transform_func)
            dict_dataset[source_localisation] = dt
            
            for key in dataset.keys():
                if key!=source_localisation:
                    x = dataset[key]["input"]
                    y = dataset[key]["label"]
                    y = relabel(y)
                    dt = tf.data.Dataset.from_tensor_slices({'image':x,'label':y})
                    dt = dt.apply(tf.data.experimental.assert_cardinality(len(x)))
                    dt = dt.map(transform_func)
                    dict_dataset[key] = dt
            return dict_dataset