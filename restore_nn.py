import numpy as np
import tensorflow as tf
import problem_domain
import neural_net
from tensorflow.python.tools import inspect_checkpoint as chkp

tf.reset_default_graph()
batch_size=1

p=problem_domain.poisson_box(batch_size)
nn=neural_net.mlp_poisson(p.x,n_layers=3,n_neurons=40)
saver=tf.train.Saver()
sess=tf.Session()

saver.restore(sess,"models/model_poisson.ckpt")
#chkp.print_tensors_in_checkpoint_file("models/model_poisson.ckpt",tensor_name='', all_tensors=True)
loss=p.get_loss(nn)
data_point=np.array([[0.8,0.6],[0.8,-0.6]])
#loss_val=sess.run(loss,{p.x:p.get_sample_random()})
print(sess.run(nn.u,{p.x:data_point}))
print(sess.run(p.source,{p.x:data_point}))
print(sess.run(nn.lap,{p.x:data_point}))
print(sess.run(loss,{p.x:data_point}))

#print(loss_val)
