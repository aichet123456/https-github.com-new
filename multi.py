import matplotlib.pyplot as plt
import numpy as np

def featureNormalization(x):
    x_np=np.asarray(x)
    mean=np.mean(x)
    std=np.std(x)
    return (x_np-mean)/std


def h_function(theta,x):
    h=np.dot(x,theta)
    return h

def grad(x,y,theta):
    m = y.size
    n = x[0, :].size
    g_vector = np.zeros(n)
    for j in range(n):
        g = 0
        for i in range(m):
            t = (h_function(theta, x[i,:]) - y[i]) * x[i,j]
            g = g + t
        g_vector[j] = (1/m)*g
    return g_vector
def gradientDescent(x,y,theta,alpha,num_iters):
    theta=np.asarray(theta)
    j_his=[cost_function(x,y,theta)]
    for i in range (num_iters):
        cost = cost_function(x,y,theta)
        gr = grad(x,y,theta)
        theta = theta - (alpha * gr)
        j_his.append(cost)
    return theta,j_his
def cost_function(x,y,theta):
    n=y.size
    cost=0
    for i in range (n):
        cost=cost+(h_function(theta,x[i,:])-y[i])**2
    return cost/(2*n)
def main():
    filename="data.txt"
    data= np.loadtxt(filename,delimiter=" ")
    x=np.c_[data[:,0:-1]]
    y=np.c_[data[:,-1]]
    #x=featureNormalization(x)
    x=np.c_[np.ones(data.shape[0]), x]
    i_theta=[0,1]
    alpha=0.01
    num=100
    theta,cost=gradientDescent(x,y,i_theta,alpha,num)
    print(theta)
    print(theta[0]+4*theta[1])# predict
    #print(cost)
    plt.figure(1)
    plt.plot(cost)
    plt.show()
if __name__=="__main__":
    main()