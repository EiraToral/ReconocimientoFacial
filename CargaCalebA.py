import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime

np.set_printoptions(precision=4)

#Esto cambia el doble espacio entre los datos de la tabla de CalebA
with open('list_attr_celeba.txt', 'r') as f:

    print("skipping : " + f.readline())
    print("skipping headers : " + f.readline())
    with open('attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            newf.write(new_line)
            newf.write('\n')