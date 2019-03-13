import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import neural_net
import problem_domain

#This script trains the neural network for solving a given pde problem
batch_size=1024
n_epochs=50
sample_type='grid'
#sample_type='random'

#Create an object of the problem class
p=problem_domain.poisson_box(batch_size)   

#Create an object of the neural network class with inputs as the data attribute(x) of the problem class
nn=neural_net.mlp_poisson(p.x,n_layers=3,n_neurons=40)

#Get loss terms
loss=p.get_loss(nn)

#Define optimizer and train operations
global_step=tf.Variable(0,trainable=True)
#start_lr=0.001
#lr=tf.train.exponential_decay(start_lr,global_step,decay_steps=5000,decay_rate=0.9,staircase=True)
lr=tf.placeholder(dtype=tf.float32,shape=[])
optimizer=tf.train.AdamOptimizer(lr)
train=optimizer.minimize(loss)

#Create a Session operation
sess=tf.Session()
#Create a saver
saver=tf.train.Saver()
#This step should be done only before starting the training loop, which initializes all the variables in the 
#tensorflow computational graph created so far.
init=tf.global_variables_initializer()
sess.run(init)

#Training loop: We do not need a predefined dataset for solving a pde. We just sample points from the domain.
#The loop generates n_batches and trains each batch for n_epochs

lr_now=0.001
cntr=1
best_loss=1e6
noimprov_cntr=0
stop_cntr=0
noimprov_tolerance=5
#Create a validation object of the problem class
val_batch_size=5000
p_val=problem_domain.poisson_box(val_batch_size)
val_data=p_val.get_sample_random()

batch_loss_history=[]
val_loss_history=[]

if sample_type=='grid':
    training_grid_data=p.get_sample_grid()
    n_batches=int(len(training_grid_data)/batch_size)
    for epochs in range(n_epochs):
         
        if val_loss_now<1e-7:
            print('Breaking, adequately low loss')
            break
        if val_loss_now<best_loss:
            best_loss=val_loss_now
        else:
            noimprov_cntr=noimprov_cntr+1
            stop_cntr=stop_cntr+1
        if noimprov_cntr>noimprov_tolerance:
            lr_now=max(lr_now/2,1e-6)
            noimprov_cntr=0
        if stop_cntr>100:
            print("Val loss: ", val_loss_now, "\n")
            print('Breaking, no loss improvement')
            break
    
        for batch in range(n_batches):
            training_data_now=training_grid_data[batch*batch_size:min((batch+1)*batch_size,len(training_grid_data)),:]
            sess.run(train,{p.x:training_data_now,lr:lr_now})
            batch_loss_now=sess.run(loss,{p.x:training_data_now})
            batch_loss_history.append(batch_loss_now)
            val_loss_now=sess.run(loss,{p.x:val_data})
            val_loss_history.append(val_loss_now)
            sys.stdout.write("Training Batch Loss: %f  Learning rate: %f val_loss: %f\r" % (batch_loss_now,lr_now,val_loss_now))
            sys.stdout.flush()
       

#if sample_type=='random':
n_batches=20
for _ in range(n_batches):
    training_data=p.get_sample_random()
    for epochs in range(n_epochs):
        sess.run(train,{p.x:training_data,lr:lr_now})
        batch_loss_now=sess.run(loss,{p.x:training_data})
        batch_loss_history.append(batch_loss_now)
        val_loss_now=sess.run(loss,{p.x:val_data})
        val_loss_history.append(val_loss_now)
        sys.stdout.write("Training Batch Loss: %f  Learning rate: %f val_loss: %f\r" % (batch_loss_now,lr_now,val_loss_now))
        sys.stdout.flush()
    if val_loss_now<1e-7:
        break

        
        
#print("Test_loss - ", sess.run(loss,p.get_sample()), "\n")
#save_path=saver.save(sess,"models/model_"+"poisson"+".ckpt")   
save_path=saver.save(sess,"./models/model_"+"poisson"+".ckpt")   

#Plotting routine
plot_variables=p.generate_plot(nn,sess)
print("Plot Loss -", sess.run(loss,{p.x:plot_variables[3]}))

with open('plot_variables.pickle','wb') as f:
    pickle.dump(plot_variables,f)

fig=plt.figure(29)
p=plt.plot(batch_loss_history,'r',val_loss_history,'b',label=['Training Loss','Validation Loss'])
plt.legend()
plt.show()


    
        
    
