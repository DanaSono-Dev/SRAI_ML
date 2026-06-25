#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:29:54 2026

@author: multi4090
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.getenv('BASE_DIR', str(Path(__file__).resolve().parent))
MODEL_DATE = os.getenv('MODEL_DATE', '2026-03-11')

path = os.path.join(BASE_DIR, MODEL_DATE, 'pltimagenes', 'datos', 'metricas.csv')
DB = np.genfromtxt(path, delimiter=',')[1:]

epoch    = DB[:,0]
loss     = DB[:,1]
acc      = DB[:,2]
val_loss = DB[:,3]
val_acc  = DB[:,4]

plt.figure(figsize=(6, 3))
plt.plot(epoch, loss, label='Loss')
plt.plot(epoch, val_loss, label='Val_loss')

plt.plot(epoch, acc, label='Acc')
plt.plot(epoch, val_acc, label='Val_acc')

plt.ylim(-0.1, 2)
plt.legend()
plt.show()
plt.close()
