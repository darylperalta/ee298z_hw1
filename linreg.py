import sys
import numpy as np

coeff = sys.argv[1:]
coeff = list(map(float,coeff))

degree = len(coeff)-1
print("degree")
print(degree)

print("coeff")
print(coeff)

#convert to Numpy array the coefficients
x = np.array(coeff)
print("x")
print(x)
x = x.reshape((-1,1))

#create input vector
num_train = 1000
val = np.array(list(range(-num_train//2,num_train//2)))
val = val/50
#val = val.reshape((-1,1))
print("val")
print(val)
print(val.shape)

A = np.zeros((num_train,degree+1))
#create A Matrix
for i in range(degree):
    A[:,i] = val**(degree-i)
A[:,degree] = np.ones((num_train))
print("A")
print(A.shape)
print(A)
#Solve for b
b = np.matmul(A,x)
print("b")
print(b)

#create noise
noise = np.random.uniform(low=-0.5,high=0.5,size=(num_train,1))
print("noise")
print(noise)
#A_noise= noise+A
#print("A with noise")
#print(A_noise)
b_noise = b+ noise
#Loss

loss = np.mean((np.matmul(A,x)-b_noise)**2)
#print(np.sum((A*x-b)**2))
#print("loss")
#print(loss)

lr = 0.00001
tol = 3
#print(np.matmul(A_noise.T,b))
#print(np.matmul(A_noise.T,np.matmul(A_noise,x)))
grad = np.matmul(A.T,np.matmul(A,x))-np.matmul(A.T,b_noise)
print("initial grad")
print(grad)
#print(np.matmul(A.T,np.matmul(A,x))-np.matmul(A.T,b))
#print(np.sqrt(np.sum(grad**2)))
#x_solved = x+np.random.uniform(low=0,high=1,size=x.shape)
x_solved =np.random.uniform(low=0,high=0.1,size=x.shape)
#x_solved = x
print("initial x")
print(x_solved)
#print(np.sqrt(np.mean(grad**2)))
print("init loss")
loss = np.mean((np.matmul(A,x_solved)-b_noise)**2)
print(loss)
iter_ctr = 0
lr_ctr = 1
#print()
#while(np.sqrt(np.mean(grad**2))>tol):

while(np.linalg.norm(grad,2)>tol):
#while(loss>tol):
    grad = np.matmul(np.matmul(A.T,A),x_solved)-np.matmul(A.T,b_noise)

    #print(grad)
    #print((lr/num_train)* grad)
    x_solved = x_solved - (lr/num_train)* grad
    loss = np.mean((np.matmul(A,x_solved)-b_noise)**2)
    #print("loss")
    #print(loss)

    #print("x")
    #print(x_solved)
    #print(np.sqrt(np.mean(grad**2)))
    print("grad norm", np.linalg.norm(grad,2), "loss", loss, "x", x_solved.flatten(), "iter", iter_ctr)
    #print(np.linalg.norm(grad,2))
    #print(iter_ctr)
    iter_ctr += 1

    '''
    if iter_ctr == lr_ctr*1000:
        lr = lr/10
    '''
print("Finished.")
#print(iter_ctr)
