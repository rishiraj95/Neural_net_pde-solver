import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class poisson_box:
    
    def __init__(self,batch_size):
        
        #Input dimension
        self.dim=2
            #Assign location for the inputs of possible shape (batch_size,input_dim)
        self.x=tf.placeholder(dtype=tf.float32, shape=[None,self.dim]) 
        #Define the source term for the problem
        self.source=-tf.reduce_sum(tf.square(self.x),axis=1)
        #Assign batch size to a class attribute
        self.batch_size=batch_size
        #Assign boundary values for the box domain
        self.xmin=-1
        self.xmax=1
        self.ymin=-1
        self.ymax=1
        #Scale is used to generate random samples defined in get_sample()
        self.xscale=self.xmax-self.xmin
        self.yscale=self.ymax-self.ymin
        #batch_ratio represents the ratio between number of boundary and interior points in a batch 
        self.batch_ratio=0.125
        #This stores the loss terms for each category of points in a sample
        self.loss_terms=[]
        #Resolution for mesh
        self.mesh_res=256
        
    def get_sample_random(self):
        #This function returns a batch of sample points for this problem
        #We pick random points from the interior and the boundaries
        x_interior=(self.xmin+np.random.rand(int(4*self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_interior=(self.ymin+np.random.rand(int(4*self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_interior=np.concatenate((x_interior,y_interior),axis=1)
        
        x_leftb=(self.xmin+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        y_leftb=(self.ymin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_leftb=np.concatenate((x_leftb,y_leftb),axis=1)
        
        x_rightb=(self.xmax+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        y_rightb=(self.ymin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_rightb=np.concatenate((x_rightb,y_rightb),axis=1)
        
        x_bottomb=(self.xmin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_bottomb=(self.ymin+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        sample_bottomb=np.concatenate((x_bottomb,y_bottomb),axis=1)
        
        x_upperb=(self.xmin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_upperb=(self.ymax+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        sample_upperb=np.concatenate((x_upperb,y_upperb),axis=1)
        
        #We put all generated points into one array with shape (batch_size,input_dim)
        data_points=np.concatenate((sample_interior,sample_leftb,sample_rightb,sample_bottomb,
                                    sample_upperb),axis=0)
        
        #Shuffle the points before returning
        np.random.shuffle(data_points)
        
        return data_points
        
    def get_sample_grid(self):
        self.mesh_res_factor=1
        self.xgrid=np.linspace(self.xmin,self.xmax,int(self.mesh_res_factor*self.mesh_res))
        self.ygrid=np.linspace(self.ymin,self.ymax,int(self.mesh_res_factor*self.mesh_res))
        self.xx,self.yy= np.meshgrid(self.xgrid,self.ygrid)
        self.data_x=self.xx.reshape(-1,1)
        self.data_y=self.yy.reshape(-1,1)
        self.data_points=np.concatenate((self.data_x,self.data_y),axis=1)
        np.random.shuffle(self.data_points)
        return self.data_points
                
    def add_loss_terms(self,condition,loss):
        #Adds loss terms where condition is true and appends to the loss_terms list
        self.loss_terms.append(
            tf.reduce_mean(
                tf.where(condition,
                         loss,
                         tf.zeros_like(loss))))
        
    def get_interior(self):
        #Returns boolean tensor with True for interior points
        return tf.logical_and(tf.logical_and(tf.less(self.x[:,0],self.xmax),
                          tf.greater(self.x[:,0],self.xmin)),tf.logical_and(tf.less(self.x[:,1],self.ymax),
                          tf.greater(self.x[:,1],self.ymin)))
    def get_leftb(self):
        #Returns boolean tensor with True for left boundary points
        return tf.equal(self.x[:,0],self.xmin)
    def get_rightb(self):
        #Returns boolean tensor with True for right boundary points
        return tf.equal(self.x[:,0],self.xmax)
    def get_bottomb(self):
        #Returns boolean tensor with True for bottom boundary points
        return tf.equal(self.x[:,1],self.ymin)
    def get_upperb(self):
        #Returns boolean tensor with True for top boundary points
        return tf.equal(self.x[:,1],self.ymax)

    
    def get_loss(self,nn):
        #Calls add_loss_terms() for different category of points and returns the sum of all loss terms
        self.add_loss_terms(self.get_interior(),loss=tf.square(nn.lap-self.source))
        self.add_loss_terms(self.get_leftb(),loss=tf.square(nn.u-tf.zeros_like(nn.u)))
        self.add_loss_terms(self.get_rightb(),loss=tf.square(nn.u-tf.zeros_like(nn.u)))
        self.add_loss_terms(self.get_bottomb(),loss=tf.square(nn.u-tf.zeros_like(nn.u)))
        self.add_loss_terms(self.get_upperb(),loss=tf.square(nn.u-tf.zeros_like(nn.u)))
        return tf.add_n(self.loss_terms)
        
    def generate_plot(self,nn,sess):
        self.xgrid=np.linspace(self.xmin,self.xmax,self.mesh_res)
        self.ygrid=np.linspace(self.ymin,self.ymax,self.mesh_res)
        self.xx,self.yy= np.meshgrid(self.xgrid,self.ygrid)
        self.plot_x=self.xx.reshape(-1,1)
        self.plot_y=self.yy.reshape(-1,1)
        self.plot_data=np.concatenate((self.plot_x,self.plot_y),axis=1)
        self.plot_u=sess.run(nn.u,{self.x:self.plot_data})
        self.plot_u=self.plot_u.reshape(self.xx.shape)
        return (self.xx,self.yy,self.plot_u,self.plot_data)
       
class stokes_box:
    
    def __init__(self,batch_size):
        
        #Input dimension
        self.dim=2
        #Assign location for the inputs of possible shape (batch_size,input_dim)
        self.x=tf.placeholder(dtype=tf.float32, shape=[None,self.dim]) 
        #Define the source term for the problem
        self.source=tf.zeros_like(self.x[:,0])
        #Assign batch size to a class attribute
        self.batch_size=batch_size
        #Assign boundary values for the box domain
        self.xmin=-0.5
        self.xmax=1
        self.ymin=-0.5
        self.ymax=1.5
        #Scale is used to genereate random samples defined in get_sample()
        self.xscale=self.xmax-self.xmin
        self.yscale=self.ymax-self.ymin
        #batch_ratio represents the ratio between number of boundary and interior points in a batch 
        self.batch_ratio=0.125
        #This stores the loss terms for each category of points in a sample
        self.loss_terms=[]
        #Resolution for mesh
        self.mesh_res=256
        
    def get_sample(self):
        #This function returns a batch of sample points for this problem
        #We pick random points from the interior and the boundaries
        x_interior=(self.xmin+np.random.rand(int(4*self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_interior=(self.ymin+np.random.rand(int(4*self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_interior=np.concatenate((x_interior,y_interior),axis=1)
        
        x_leftb=(self.xmin+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        y_leftb=(self.ymin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_leftb=np.concatenate((x_leftb,y_leftb),axis=1)
        
        x_rightb=(self.xmax+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        y_rightb=(self.ymin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.yscale).reshape(-1,1)
        sample_rightb=np.concatenate((x_rightb,y_rightb),axis=1)
        
        x_bottomb=(self.xmin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_bottomb=(self.ymin+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        sample_bottomb=np.concatenate((x_bottomb,y_bottomb),axis=1)
        
        x_upperb=(self.xmin+np.random.rand(int(self.batch_ratio*self.batch_size))*self.xscale).reshape(-1,1)
        y_upperb=(self.ymax+np.zeros(int(self.batch_ratio*self.batch_size))).reshape(-1,1)
        sample_upperb=np.concatenate((x_upperb,y_upperb),axis=1)
        
        #We put all generated points into one array with shape (batch_size,input_dim)
        data_points=np.concatenate((sample_interior,sample_leftb,sample_rightb,sample_bottomb,
                                    sample_upperb),axis=0)
        
        #Shuffle the points before returning
        np.random.shuffle(data_points)
        
        return {self.x:data_points}
        
   
    def add_loss_terms(self,condition,loss):
        #Adds loss terms where condition is true and appends to the loss_terms list
        self.loss_terms.append(
            tf.reduce_mean(
                tf.where(condition,
                         loss,
                         tf.zeros_like(loss))))
        
    def get_interior(self):
        #Returns boolean tensor with True for interior points
        return tf.logical_and(tf.logical_and(tf.less(self.x[:,0],self.xmax),
                          tf.greater(self.x[:,0],self.xmin)),tf.logical_and(tf.less(self.x[:,1],self.ymax),
                          tf.greater(self.x[:,1],self.ymin)))
    def get_leftb(self):
        #Returns boolean tensor with True for left boundary points
        return tf.equal(self.x[:,0],self.xmin)
    def get_rightb(self):
        #Returns boolean tensor with True for right boundary points
        return tf.equal(self.x[:,0],self.xmax)
    def get_bottomb(self):
        #Returns boolean tensor with True for bottom boundary points
        return tf.equal(self.x[:,1],self.ymin)
    def get_upperb(self):
        #Returns boolean tensor with True for top boundary points
        return tf.equal(self.x[:,1],self.ymax)

    
    def get_loss(self,nn):
        #Calls add_loss_terms() for different category of points and returns the sum of all loss terms
        self.nu=0.025
        self.eta=1/(2*self.nu)-np.sqrt(1/(4*self.nu**2)+4*np.pi**2)
        
        self.interior_loss=tf.multiply(nn.psix[:,1],nn.omegax[:,0])
        -tf.multiply(nn.psix[:,0],nn.omegax[:,1])-self.nu*nn.lap_omega

        self.boundary_loss=nn.psi-(self.x[:,1]-1/(2*np.pi)
                                   *tf.multiply(tf.exp(self.eta*self.x[:,0]),tf.sin(2*np.pi*self.x[:,1])))
        
        self.add_loss_terms(self.get_interior(),loss=tf.square(self.interior_loss))
        self.add_loss_terms(self.get_leftb(),loss=tf.square(self.boundary_loss))
        self.add_loss_terms(self.get_rightb(),loss=tf.square(self.boundary_loss))
        self.add_loss_terms(self.get_bottomb(),loss=tf.square(self.boundary_loss))
        self.add_loss_terms(self.get_upperb(),loss=tf.square(self.boundary_loss))
        return tf.add_n(self.loss_terms)
        
    def generate_plot(self,nn,sess):
        self.xgrid=np.linspace(self.xmin,self.xmax,self.mesh_res)
        self.ygrid=np.linspace(self.ymin,self.ymax,self.mesh_res)
        self.xx,self.yy= np.meshgrid(self.xgrid,self.ygrid)
        self.plot_x=self.xx.reshape(-1,1)
        self.plot_y=self.yy.reshape(-1,1)
        self.plot_data=np.concatenate((self.plot_x,self.plot_y),axis=1)
        self.plot_psi=sess.run(nn.psi,{self.x:self.plot_data})
        self.plot_psix=sess.run(nn.psix,{self.x:self.plot_data})
        self.plot_u=self.plot_psix[:,1]
        self.plot_v=-self.plot_psix[:,0]
        self.plot_energy=np.square(self.plot_u)+np.square(self.plot_v)
        self.plot_energy=self.plot_energy.reshape(self.xx.shape)
        self.plot_psi=self.plot_psi.reshape(self.xx.shape)
        plt.figure()
        plt.pcolormesh(self.xx,self.yy,self.plot_energy,cmap='hot')
        print(self.plot_psi,"\n")
        plt.show()
        return self.plot_data
                

                
        
