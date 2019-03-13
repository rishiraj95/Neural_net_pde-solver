import sys
import numpy as np
import tensorflow as tf

class mlp_poisson:
    #lap(u)=f
    def __init__(self,inputs=[],n_layers=1,n_neurons=10,output_dim=1):
        
        self.inputs=inputs
        self.n_layers=n_layers
        self.n_neurons=n_neurons
        self.output_dim=output_dim
        
        
        #First hidden layer
        self.hidden_layers=[tf.layers.dense(self.inputs,self.n_neurons,activation=tf.nn.tanh)]
        #Subsequent hidden layers
        for i in range(1,self.n_layers):
            self.hidden_layers.append(tf.layers.dense(self.hidden_layers[i-1],self.n_neurons,activation=tf.nn.tanh))
            
        #Calculate the output and the required derivatives as needed in the problem
        self.output=tf.layers.dense(self.hidden_layers[-1],self.output_dim)
        self.u=tf.squeeze(self.output)
        self.ux=tf.gradients(self.u,self.inputs)[0]
        self.uxx=tf.transpose([tf.gradients(self.ux[:,i],self.inputs)[0][:,i] for i in range(self.inputs.shape[1])])
       # self.uxx=tf.gradients(self.ux,self.inputs)[0]
        self.lap=tf.reduce_sum(self.uxx,axis=1)


class mlp_stokes:
    
    def __init__(self,inputs=[],n_layers=1,n_neurons=10,output_dim=1):
        
        self.inputs=inputs
        self.n_layers=n_layers
        self.n_neurons=n_neurons
        self.output_dim=output_dim
        
        
        #First hidden layer
        self.hidden_layers=[tf.layers.dense(self.inputs,self.n_neurons,activation=tf.nn.tanh)]
        #Subsequent hidden layers
        for i in range(1,self.n_layers):
            self.hidden_layers.append(tf.layers.dense(self.hidden_layers[i-1],self.n_neurons,activation=tf.nn.tanh))
            
        #Calculate the output and the required derivatives as needed in the problem
        self.output=tf.layers.dense(self.hidden_layers[-1],self.output_dim)
        self.psi=tf.squeeze(self.output)
        self.psix=tf.gradients(self.psi,self.inputs)[0]
        self.psixx=tf.gradients(tf.gradients(self.psi,self.inputs)[0],self.inputs)[0]
        self.omega=-tf.reduce_sum(self.psixx,axis=1)
        self.omegax=tf.gradients(self.omega,self.inputs)[0]
        self.omegaxx=tf.gradients(self.omegax,self.inputs)[0]
        self.lap_omega=tf.reduce_sum(self.omegaxx,axis=1)
