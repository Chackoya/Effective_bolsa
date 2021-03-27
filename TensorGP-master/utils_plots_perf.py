#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:33:25 2020

@author: gama
"""
import matplotlib.pyplot as plt

def plotLOSS(history_model,export_path,prefix):
    plt.figure(figsize=[8,6])

    #print(history_model.history['loss'])
    #print(history_model.history['val_loss'])
    plt.plot(history_model.history['loss'],'r',linewidth=3.0)
    plt.plot(history_model.history['val_loss'],'b',linewidth=3.0)
    #plt.ylim((0.001,0.1))
    plt.legend(['Training loss', 'Validation Loss'],fontsize=10, loc='best')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    #f.axes.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title('Loss Curves',fontsize=16)    
    #plt.show(block=False)

    plt.savefig(export_path+'/'+ prefix+'_loss.png')


def plotACCURACY(history_model,export_path, prefix):
  # Accuracy Curves
    plt.figure(figsize=[8,6])
    #plt.plot(history_model.history['binary_accuracy'],'r',linewidth=3.0)
    #plt.plot(history_model.history['val_binary_accuracy'],'b',linewidth=3.0)
    
    plt.plot(history_model.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history_model.history['val_accuracy'],'b',linewidth=3.0)
    plt.ylim((0.50,1.0))
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10,loc='best')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    #f.axes.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title('Accuracy Curves',fontsize=16)
    #plt.show(block=False)
    
    plt.savefig(export_path+'/'+ prefix+'_acc.png')