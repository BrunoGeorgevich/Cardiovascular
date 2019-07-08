#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:13:55 2019

@author: bruno
"""

from statsmodels.nonparametric.smoothers_lowess import lowess

from scipy.interpolate import interp1d

from keras.optimizers import SGD
from keras.utils import to_categorical
import keras

import matplotlib.pyplot as plt
import pandas as pd

from numpy.random import randn
import numpy as np
import time

INITIAL_HR = 60
HR = INITIAL_HR
RR = 60/HR

PAD = 15
ONE_S = 360
WINDOW_SIZE = 3
STEP  = int(4*ONE_S)
NORM_STEP = int(ONE_S/2)

def normalize(arr):
    g = max(abs(max(arr)),abs(min(arr)))
    return g

def fit(x, y, deg):    
    params = np.polyfit(x, y, deg)
    baseline = np.polyval(params, x)
    return baseline

def mean_average_filter(arr, ws):
    return np.array(list(pd.DataFrame(arr)[0].rolling(ws).mean()))

def preprocess_data(folder,file):
    initial_time = time.time()

    df = pd.read_csv(f'database/preprocessed_data/{folder}/{file}.csv')

    new_df = pd.DataFrame()
    
    del df[df.keys()[0]]
    del df[df.keys()[1]]

    prelabel = pd.read_fwf(f'database/preprocessed_data/{folder}/{file}annotations.txt')
    prelabel = prelabel.loc[prelabel['Type']  == 'N']
    
    no_baseline_data = np.array([])
    normalize_signal = np.array([])
    
    len_data = len(df.values[:,0])
    for index in range(0, np.ceil(len_data/STEP).astype(int)):
        before = STEP*index
        after  = len_data if STEP*(index+1) > len_data else STEP*(index+1)
    
        data = np.copy(df.values[before:after,0])
    
        t = [i for i in range(0, len(data)) ]
        z = lowess(data, t, 0.138888889, WINDOW_SIZE)[:,1]
    
        baseline_removed_data = data - z
    
        norm_sample = np.zeros_like(data)
    
        for row in range(NORM_STEP,len(data), NORM_STEP*2):
            temp_data = baseline_removed_data[row - NORM_STEP:row + NORM_STEP]
            norm_sample[row + (np.argmax(np.absolute(temp_data)) - NORM_STEP)] = normalize(temp_data)
    
    
        no_baseline_data = np.append(no_baseline_data,baseline_removed_data)
        normalize_signal = np.append(normalize_signal,norm_sample)
    
    normalize_signal = np.array([(i,x) for i,x in enumerate(normalize_signal) if x])
    
    x = np.array([i for i in range(len(no_baseline_data))])
    g = interp1d(normalize_signal[:,0],normalize_signal[:,1], fill_value='extrapolate')
    g = g(x)
    
    normalized_no_baseline_data = no_baseline_data/g
    
    new_df['data'] = normalized_no_baseline_data
    del df[df.keys()[0]]
    
    label = np.zeros_like(normalized_no_baseline_data)
    
    for i,row in enumerate(prelabel.values[:,1]):
        label[row - 15: row  + 15] = 1
    
    new_df['label'] = label
    
    with open(f'database/processed_data/{folder}/{file}.csv','w') as f:
        f.write(new_df.to_csv(index=False))
              
    end_time = time.time()
    
    print(f"Ellapsed Time: {end_time - initial_time}")

def posprocess_data(folder, file):
    initial_time = time.time()

    df = pd.read_csv(f'database/processed_data/{folder}/{file}.csv')
    data = df.values[:,0]

    labels = pd.read_fwf(f'database/preprocessed_data/{folder}/{file}annotations.txt')
    labels = labels.loc[labels['Type']  == 'N'].values[:,1]

    begin_arr = []
    end_arr = []
    new_label = []
    
    for i in range(0, len(data) - 145):
        begin = i
        end = i + 145
    
        temp_label = 0
    
        for label in labels:
            if begin > (label + PAD):
                continue
            if end < (label - PAD):
                continue
            if begin + 36 in range(label - PAD, label + PAD):
                temp_label = 1
            break
    
        new_label.append(temp_label)
        begin_arr.append(begin)
        end_arr.append(end)


    new_label = np.array(new_label).flatten()
    begin_arr = np.array(begin_arr).flatten()
    end_arr = np.array(end_arr).flatten()

    label_df = pd.DataFrame()
    label_df['label'] = new_label
    label_df['begin'] = begin_arr
    label_df['end'] = end_arr
        
    with open(f'database/posprocessed_data/{folder}/label_{file}.csv','w') as f:
        f.write(label_df.to_csv(index=False))
    
    end_time = time.time()

    print(f"Ellapsed Time: {end_time - initial_time}")

def translate_label(val, data, size=145):
    label = [0 for i in range(145)]

    if val == 1:
        label[36] = 1

    return label

def retrieve_labels_predict(test_data):
    test_data = test_data.reshape((-1,145,1))
    predict = model.predict(test_data)[0]
    test_data = test_data.reshape(145)

    if predict[1] > predict[0] and test_data[36] == 1:
        return test_data[36]

    return None

def remove_baseline_normalize(input_data):
    no_baseline_data = np.array([])
    normalize_signal = np.array([])
    
    len_data = len(input_data)
    for index in range(0, np.ceil(len_data/STEP).astype(int)):
        before = STEP*index
        after  = len_data if STEP*(index+1) > len_data else STEP*(index+1)
    
        data = np.copy(input_data[before:after])
    
        t = [i for i in range(0, len(data)) ]
        z = lowess(data, t, 0.138888889, WINDOW_SIZE)[:,1]
    
        baseline_removed_data = data - z
    
        norm_sample = np.zeros_like(data)
    
        for row in range(NORM_STEP,len(data), NORM_STEP*2):
            temp_data = baseline_removed_data[row - NORM_STEP:row + NORM_STEP]
            norm_sample[row + (np.argmax(np.absolute(temp_data)) - NORM_STEP)] = normalize(temp_data)
    
    
        no_baseline_data = np.append(no_baseline_data,baseline_removed_data)
        normalize_signal = np.append(normalize_signal,norm_sample)
    
    normalize_signal = np.array([(i,x) for i,x in enumerate(normalize_signal) if x])
    
    x = np.array([i for i in range(len(no_baseline_data))])
    g = interp1d(normalize_signal[:,0],normalize_signal[:,1], fill_value='extrapolate')
    g = g(x)
    
    return no_baseline_data/g

def complete_w_zeros(data, total_size = 145):
    data_len = len(data)
    needed = total_size - data_len
    data = np.copy(data)

    for i in range(needed):
        data = np.append(data,0)
    return data

def extrapolate(data, total_size = 145):
    data_len = len(data)
    needed = total_size - data_len
    data = np.copy(data)

    last = data[-1]
    for i in range(needed):
        last += 1
        data = np.append(data,last)
    return data

def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
	    )( dt * f( t + dt/2, y + dy2/2 ) )
	    )( dt * f( t + dt/2, y + dy1/2 ) )
	    )( dt * f( t       , y         ) )

def iterative_func(t,x):
    global z0
    x = np.array(x)
    A_ecg = [1.2, -5, 30, -7.5, 0.75]
    B_ecg = [0.25, 0.1, 0.1, 0.1, 0.4]
    TH = np.array([-1/3, -1/12, 0, 1/12, 1/2])*np.pi
    w = 2*np.pi/RR
    k = 0
    alpha = 1 - np.sqrt(x[0]**2 + x[1]**2)
    th = np.arctan2(x[1],x[0])
    
    for i in range(0,5):
        k -= (A_ecg[i]*(th-TH[i])*np.exp((-(th - TH[i])**2)/(2*B_ecg[i]**2)))
    
    return np.array([
                alpha*x[0] -     w*x[1] + 0*x[2],
                    w*x[0] - alpha*x[1] + 0*x[2],
                    0*x[0] -     0*x[1] - 1*x[2] + k + z0[-1]
           ])

def mamemi(zecg):
    N = len(zecg)
    maxi = np.zeros(N)
    mini = np.zeros(N)
    h = np.zeros(N)
    a = np.zeros(N)
    n = np.zeros(N)
    g = np.zeros(N)
    r = np.zeros(N)
    v = np.zeros(N)
    w = np.zeros(N)
    R_dtct = np.zeros(N)
    
    deltaS = np.zeros(N)
    deltaS[0] = 1
    
    sigma = 2
    delta = 2
    beta = 15
    atv = 0
    for i in range(N - 1):
        if zecg[i + 1] > maxi[i]:
            maxi[i + 1] = maxi[i] + sigma*delta
        elif zecg[i + 1] <= maxi[i]:
            maxi[i + 1] = maxi[i] - delta
    
        if zecg[i + 1] < mini[i]:
            mini[i + 1] = mini[i] - sigma*delta
        elif zecg[i + 1] >= mini[i]:
            mini[i + 1] = mini[i] + delta
    
        h[i+1] = zecg[i+1] - (maxi[i+1] + mini[i+1])/2
        a[i+1] = maxi[i+1] - mini[i+1]
    
        if a[i+1] <= h[i+1]:
            n[i+1] = np.sign(h[i+1]*(abs(h[i+1]) - a[i+1]))
    
        if i > beta:
            if (n[i] > 0) and (n[i] > n[i-beta]) and (n[i] > n[i+beta]):
                g[i] = n[i] - max(n[i-beta],n[i+beta])
            elif (n[i] < 0) and (n[i] < n[i-beta]) and (n[i] < n[i+beta]):
                g[i] = n[i] + min(n[i-beta],n[i+beta])
    
            if g[i] > g[i - 1] and g[i] > g[i + 1]:
                r[i] = g[i]
    
            if g[i] < g[i - 1] and g[i] < g[i + 1]:
                v[i] = g[i]
    
        if r[i] > 0:
            w[i] = r[i]
    
        if v[i] < 0:
            w[i] = -v[i]
    
        if w[i] == 1:
            atv = 1
    
        if (atv == 1) and (zecg[i] >= 0.03):
            atv = 2
    
        if (atv == 2) and (zecg[i] <= 0.03):
            atv = 0
            R_dtct[i] = 1
    return R_dtct

def generate_synthetic_ecg(duration):
    global RR
    global HR
    global z0

    start_t = 0
    ts = 0.002777778
    end_t = duration
    
    T = np.arange(start_t, end_t, ts)
    n = len(T)

    tmax = 0.2 + 0.1555*RR
    ts_VAD = 0.5
    t_eject = (RR*ts_VAD)/ts
    
    xecg = np.zeros(n)
    yecg = np.zeros(n)
    zecg = np.zeros(n)
    z0 = np.zeros(n)
    
    xecg[0] = -1
    
    x = [xecg[0], yecg[0], zecg[0]]
    
    ip = 0
    lg = 0
    ii = 0
    
    y = RK4(iterative_func)
    z0_dot = lambda t: 0.15*np.sin(2*np.pi*(60/(12+randn()))*t)
    
    for i in range(len(T) - 1):
        x += y(T[i + 1], x, ts)
        z0 = np.append(z0, z0_dot(T[i + 1]))
        HR = INITIAL_HR + 2*randn()
        RR = 60/HR
    
        xecg[i] = x[0]
        yecg[i] = x[1]
        zecg[i] = x[2]
    
        if i > ip:
            print(f'Executing ... {lg}')
            lg += 10
            ip += (n-1)/10

    return zecg

def load_data_labels(folder):
    data = {}
    label = {}
    
    for file in train_files:
        data_df = pd.read_csv(f'database/processed_data/{folder}/{file}.csv')
        label_df = pd.read_csv(f'database/posprocessed_data/{folder}/label_{file}.csv')
    
        data[str(file)] = np.copy(data_df['data'].values)
        label[str(file)] = np.copy(label_df.values)
    return data, label

def generator(batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 145, 1))
    batch_labels = np.zeros((batch_size,2))
    while True:
      for i in range(batch_size):
          file = np.random.choice(train_files)
        
          data = train_data[str(file)]
          label = train_label[str(file)]

          item = label[np.random.choice(len(label))]

          features = data[item[1]:item[2]]

          batch_features[i] = features.reshape((145,1))
          batch_labels[i] = to_categorical(item[0],2)
      yield batch_features, batch_labels

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(32, (5),name="C1", activation='relu', strides=(1), input_shape=(145,1,)))
    model.add(keras.layers.Dropout(0.5, name="D1"))
    model.add(keras.layers.MaxPool1D((3), (2),name="P1"))
    model.add(keras.layers.Conv1D(32, (5),name="C2", activation='relu', strides=(1)))
    model.add(keras.layers.Dropout(0.5, name="D2"))
    model.add(keras.layers.Flatten(name="Flatten"))
    model.add(keras.layers.Dense(1024, name="F1", activation='relu'))
    model.add(keras.layers.Dropout(0.5, name="D3"))
    model.add(keras.layers.Dense(512, name="F2", activation='relu'))
    model.add(keras.layers.Dropout(0.5, name="D4"))
    model.add(keras.layers.Dense(2, name="Output", activation='softmax'))
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=SGD(lr=0.0005, momentum=0.9), metrics=['accuracy'])
    return model

def predict_with_cnn(data):
    global current_time
    
    label = np.zeros(35)
    label[label == 0] = None
    
    print("PREDICTING LABELS WITH CNN 1D")
    
    step = 145
    for i in range(0, len(data) - step):
        label = np.append(label, retrieve_labels_predict(complete_w_zeros(data[i:i+step])))
    
    print(f"ELLAPSED TIME: {time.time() - current_time}")
    current_time = time.time()

    return label

def elastance():
    global HR
    Emax = 2
    Emin = 0.06
    
    tc = 60/HR
    Tmax = 0.2 + 0.15*tc
    
    #Funções
    tn = lambda t: (t % tc)/Tmax
    En = lambda t: 1.55*(((tn(t)/0.7)**1.9)/(1 + ((tn(t)/0.7)**1.9)))*((1)/(1 + ((tn(t)/1.17)**21.9)))
    E  = lambda t: (Emax - Emin)*En(t) + Emin
    
    t = np.arange(0, 60/HR, 0.002777778)
    return E(t)

def retrieve_elastance(data, label):
    e = np.zeros_like(data)
    e[e == 0] = 0.06
    
    elastance_size = len(elastance())
    for i in [index for index,val in enumerate(label) if val == 1]:
        if i + elastance_size > len(data):
            e[i:] = elastance()[:len(data) - i]
        else:
            e[i:i+elastance_size] = elastance()
    return e

def heart_parameters(E, duration):
    Rs = 1
    Rm = 0.005
    Ra = 0.001
    Rc = 0.0398
    
    Cr = 4.4
    Cs = 1.33
    Ca = 0.08
    Ls = 0.0005
    Vo = 10
    
    Dm = 1
    Da = 1

    def valves(Pae, Pao, Pve):
        if Pae > Pve:
            Da = 0
            Dm = 1
        elif Pve > Pao:
            Da = 1
            Dm = 0
        else:
            Da = 0
            Dm = 0  
        return Dm,Da
    
    A = lambda t,y: np.array([
                        [-((Dm/Rm)+(Da/Ra))*E(t), Dm/Rm, 0, Da/Ra, 0],
                        [Dm*E(t)/(Rm*Cr),-((1/Rs) + (Dm/Rm))*(1/Cr), 0,0,1/(Rs*Cr)],
                        [0,0,-Rc/Ls, 1/Ls, -1/Ls],
                        [Da*E(t)/(Ra*Ca), 0, -1/Ca, -Da/(Ra*Ca),0],
                        [0,1/(Rs*Cs),1/Cs,0,-1/(Rs*Cs)]
                     ])
    
    f = lambda t,y: np.array([
                        [((Dm/Rm) + (Da/Ra))*E(t)*Vo],
                        [-(Dm/(Rm*Cr))*E(t)*Vo],
                        [0],
                        [-(Da/(Ra*Ca))*E(t)*Vo],
                        [0]
                    ])  
        
    x_dot   = lambda t,y:A(t,y)@np.array(x) + f(t,y)
    Suga_Sagawa = lambda t:E(t)*(Vve - Vo)
    
    dy = RK4(x_dot)
    
    Vve,Pae,Qa,Pao,Ps = [ 140,   5, 0,  90,   90]
    
    x = [[Vve],[Pae],[Qa],[Pao],[Ps]]
    
    Vve_val = []
    Pve_val = []
    Pae_val = []
    Pao_val = []
    Qa_val = []
    tempo = []
    
    t, dt = 0, 0.0001
    index = 0
    while t <= duration:
        Pve = Suga_Sagawa(t)
        Dm, Da = valves(Pae, Pao, Pve)  
        
        Vve_val.append(Vve)
        Pve_val.append(Pve)
        Pae_val.append(Pae)
        Pao_val.append(Pao)
        Qa_val.append(Qa)
        tempo.append(t)
        
        t, dx = t + dt, dy(t,x,dt)
        x += dx
        index += 1
        
        Vve,Pae,Qa,Pao,Ps = x.T[0]
    return Vve_val, Pve_val, Pae_val, Pao_val, Qa_val

train_files = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
test_files = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

#%%

# PREPROCESS MIT-BIH DATASET

for file in train_files:
    preprocess_data('train',file)

for file in test_files:
    preprocess_data('test',file)
#%%

# POSPROCESS MIT-BIH DATASET

for file in train_files:
    posprocess_data('train', file)
for file in test_files:
    posprocess_data('test', file)

#%%

# TRAIN THE CNN MODEL

train_data, train_label = load_data_labels('train')
test_data, test_label = load_data_labels('test')

model = build_model()

history = model.fit_generator(generator(512),
                    steps_per_epoch=1000,
                    epochs=3)

model.save("model.h5")
#%%

# GENERATE SYNTHETIC DATA

simulation_time = 3

current_time = time.time()
initial_time = time.time()

print("GENERATING ECG")

zecg = generate_synthetic_ecg(simulation_time)

print(f"ELLAPSED TIME: {time.time() - current_time}")
current_time = time.time()

print("EXECUTING MAMEMI")

R_dtct = mamemi(zecg)
R_dtct[R_dtct == 0] = None
data = remove_baseline_normalize(zecg)

mamemi_label = np.copy(R_dtct)
mamemi_label[R_dtct == 1] = data[R_dtct == 1]

print(f"ELLAPSED TIME: {time.time() - current_time}")
current_time = time.time()

print("LOADING THE MODEL")

model = keras.models.load_model('model.h5')

print(f"ELLAPSED TIME: {time.time() - current_time}")
current_time = time.time()

label = predict_with_cnn(data)

print("GENERATING ELASTANCE CURVES")

e_cnn = retrieve_elastance(data, label)
e_mamemi = retrieve_elastance(data, R_dtct)

print(f"ELLAPSED TIME: {time.time() - current_time}")
current_time = time.time()

print("CALCULATING HEART PARAMETERS")

E_cnn = interp1d(np.arange(0,simulation_time,0.002777778),e_cnn, fill_value='extrapolate')
Vve_val_cnn, Pve_val_cnn, Pae_val_cnn, Pao_val_cnn, Qa_val_cnn = heart_parameters(E_cnn,simulation_time)
E_mamemi = interp1d(np.arange(0,simulation_time,0.002777778),e_mamemi, fill_value='extrapolate')
Vve_val_mamemi, Pve_val_mamemi, Pae_val_mamemi, Pao_val_mamemi, Qa_val_mamemi = heart_parameters(E_mamemi,simulation_time)

print(f"ELLAPSED TIME: {time.time() - current_time}")
current_time = time.time()

print(f"TOTAL ELLAPSED: {time.time() - initial_time}")
#%%

# PLOT RESULTS

plt.figure(1, figsize=(16,9))

plt.subplot(5,1,1)
plt.ylim(-1.5,3)
plt.plot(data,'green')
plt.plot(mamemi_label, 'r*')
plt.plot(label, 'bo')
plt.legend(["ECG","MaMeMi","CNN 1D"],loc='upper left')
plt.ylabel("mV")
plt.title("Sinal ECG Sintético")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.subplot(5,1,2)
plt.plot(e_mamemi,'r')
plt.plot(e_cnn,'b')
plt.legend(["MaMeMi","CNN 1D"],loc='upper left')
plt.title("Elastância")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.subplot(5,1,3)
plt.title('Simulação Hemodinâmica de um Coração Normal')
plt.ylabel('Pressões (mmHg)')
plt.plot(Pve_val_mamemi,'r')
plt.plot(Pve_val_cnn,'b')
plt.plot(Pao_val_mamemi,'r')
plt.plot(Pao_val_cnn,'b')
plt.plot(Pae_val_mamemi,'r')
plt.plot(Pae_val_cnn,'b')
plt.legend(["MaMeMi","CNN 1D"],loc='upper left')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.subplot(5,1,4)
plt.ylabel('Fluxo Aórtico (m/s)')
plt.plot(Qa_val_mamemi, 'r')
plt.plot(Qa_val_cnn, 'b')
plt.legend(["MaMeMi","CNN 1D"],loc='upper left')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.subplot(5,1,5)
plt.ylabel('Volume no VE (ml)')
plt.plot(np.arange(0,simulation_time, 0.0001), Vve_val_mamemi, 'r')
plt.plot(np.arange(0,simulation_time, 0.0001), Vve_val_cnn, 'b')
plt.legend(["MaMeMi","CNN 1D"],loc='upper left')
plt.xlabel("Tempo (s)")
plt.show()
#%%