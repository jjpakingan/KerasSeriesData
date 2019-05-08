'''
-#----#---------------------------------#####-----------------------------------######----------------------
-#---#--######-#####----##----####-----#-----#-######-#####--#-######--####-----#-----#---##---#####---##---
-#--#---#------#----#--#--#--#---------#-------#------#----#-#-#------#---------#-----#--#--#----#----#--#--
-###----#####--#----#-#----#--####------#####--#####--#----#-#-#####---####-----#-----#-#----#---#---#----#-
-#--#---#------#####--######------#----------#-#------#####--#-#-----------#----#-----#-######---#---######-
-#---#--#------#---#--#----#-#----#----#-----#-#------#---#--#-#------#----#----#-----#-#----#---#---#----#-
-#----#-######-#----#-#----#--####------#####--######-#----#-#-######--####-----######--#----#---#---#----#-

Jeff Pakingan

This code is for:
    - Experiment purpose
    - Training and testing a model based from a linear data

License: 
    https://creativecommons.org/publicdomain/zero/1.0/
'''

import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


'''
######----------------------
#-----#---##---#####---##---
#-----#--#--#----#----#--#--
#-----#-#----#---#---#----#-
#-----#-######---#---######-
#-----#-#----#---#---#----#-
######--#----#---#---#----#-
'''

X = [
    [0,1,2],
    [0,6,12],
    [0,3,6],
    [0,10.5,20],
    [0,12,23],
    [0,9,18],

    [0,2.25,3],
    [0,0,0],
    [0,11,21],
    [0,13,26],
    [0,2,5],
    [0,20,100]
    ]

X_train = np.array(X[0:6])
X_test = np.array(X[11:12])

y = [
    0,
    0,
    0,
    1, 
    1,
    1,

    0,0,1,1,0,1
]

y_train = np.array(y[0:6])
y_test = np.array(y[11:12])




'''
-#-----#--------------------######--------------------------------------------
-##---##---##---#-#----#----#-----#-#####---####---####--#####----##---#----#-
-#-#-#-#--#--#--#-##---#----#-----#-#----#-#----#-#----#-#----#--#--#--##--##-
-#--#--#-#----#-#-#-#--#----######--#----#-#----#-#------#----#-#----#-#-##-#-
-#-----#-######-#-#--#-#----#-------#####--#----#-#--###-#####--######-#----#-
-#-----#-#----#-#-#---##----#-------#---#--#----#-#----#-#---#--#----#-#----#-
-#-----#-#----#-#-#----#----#-------#----#--####---####--#----#-#----#-#----#-
'''

MODEL_PATH = 'c:\\temp\\savedmodel'
exists = os.path.isfile(MODEL_PATH)
boolForceRewriteModel = False


model = Sequential()
if(exists and not(boolForceRewriteModel)):
    model = load_model(MODEL_PATH)
else:
    model.add(Dense(6, input_shape=(3,), activation="tanh",name="1"))
    model.add(Dense(6, activation="tanh",name="2"))
    model.add(Dense(1, activation="sigmoid",name="3"))
    model.summary()
    plot_model(model, to_file = 'c:\\temp\\model.png', show_shapes = True, show_layer_names = True)
    model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
    myCallbacks = [EarlyStopping(monitor='acc',patience=30,mode=max)]
    model.fit(X_train, y_train, epochs=1000, verbose=1,callbacks = myCallbacks)
    model.save(MODEL_PATH)
eval_result = model.evaluate(X_test, y_test)
print("Test loss:", eval_result[0], "Test accuracy:", eval_result[1])
