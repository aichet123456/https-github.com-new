import numpy as np
import matplotlib.pyplot as plt

def sigmod(z):
    return (1/(1+np.exp(-z)))




def hfunction(theta, x):
    z = np.dot(theta, x)
    h = np.asarray(sigmod(z))
    return h



def gradientFunction(x, y, theta):
    m = y.size
    n = x[0, :].size
    g_vector = np.zeros(n)
    for j in range(n):
        g = 0
        for i in range(m):
            t = (hfunction(theta, x[i,:]) - y[i]) * x[i,j]
            g = g + t
        g_vector[j] = (1/m)*g
    return g_vector





def cost_function(x,y,theta):
    m=y.size
    error=0
    for i in range (m):
        t=(-y[i]*np.log(hfunction(theta,x[i,:]+1e-7)))-((1-y[i])*np.log(1-hfunction(theta,x[i,:])+1e-7))
        error+=t
    return (1/m)*error





def gradientDescent(x,y,theta,alpha,num_iters):
    #m=y.size
    theta=np.asarray(theta)
    j_his=[cost_function(x,y,theta)]
    for i in range (num_iters):
        cost = cost_function(x,y,theta)
        grad = gradientFunction(x,y,theta)
        theta = theta - (alpha * grad)
        #print(theta)
        j_his.append(cost)
    return theta,j_his



def featureNormalization(x):
    x_np=np.asarray(x)
    mean=np.mean(x)
    std=np.std(x)
    return (x_np-mean)/std





def main():
    filename="ex2data1.txt"
    data= np.loadtxt(filename,delimiter=",")
    x=np.c_[data[:,:-1]]
    x=featureNormalization(x)
    x=np.c_[np.ones(data.shape[0]), x]
    #print(x)
    y=np.c_[data[:,-1]]
    idx0= np.where(y==0)
    idx1=np.where(y==1)
    num_iters=200
    alpha=0.5
    initial_theta=[0,0,0]
    print(hfunction(initial_theta,x[0,:]))
    theta,j_his=gradientDescent(x,y,initial_theta,alpha,num_iters)
    print(theta)
    #theta,j_his=gradientDescent(x,y,initial_theta,alpha,num_iters)
    plt.figure(2)
    plt.plot(j_his)
    plt.figure(3)
    plt.scatter(x[idx1,1],x[idx1,2],s=30,c='r', marker='o')
    plt.scatter(x[idx0,1],x[idx0,2],s=30,c='b', marker='o')
    print(np.max(x[:,1]))
    x_value=np.array([np.min(x[:,1]),np.max(x[:,1])])
    y_value=-(theta[0]+theta[1]*x_value)/theta[2]
    plt.plot(x_value,y_value,"r")
    plt.show()

if __name__=="__main__":
    main()  
    