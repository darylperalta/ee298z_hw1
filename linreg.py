import sys
import numpy as np

#hyperparameters
lr = 0.00001
tol = 0.5

coeff = sys.argv[1:]
coeff = list(map(float,coeff))

degree = len(coeff)-1
print("degree")
print(degree)

print("coeff")
print(coeff)

#convert to Numpy array the coefficients
x = np.array(coeff)
x = x.reshape((-1,1))

#create input vector
num_train = 1000
val = np.array(list(range(-num_train//2,num_train//2)))
val = val/50


A = np.zeros((num_train,degree+1))
#create A Matrix
for i in range(degree):
    A[:,i] = val**(degree-i)
A[:,degree] = np.ones((num_train))

#Solve for b
b = np.matmul(A,x)

#create noise
noise = np.random.uniform(low=-1,high=1,size=(num_train,1))

b_noise = b+ noise
#Loss

loss = np.mean((np.matmul(A,x)-b_noise)**2)

grad = np.matmul(A.T,np.matmul(A,x))-np.matmul(A.T,b_noise)

x_solved =np.random.uniform(low=0,high=0.1,size=x.shape)

loss1 = np.mean((np.matmul(A,x_solved)-b_noise)**2)

iter_ctr = 0
lr_ctr = 1


while(np.linalg.norm(grad,2)>tol):
#while(loss>tol):
    loss2 = np.mean((np.matmul(A,x_solved)-b_noise)**2)
    if(loss2>loss1):
        lr = lr/5
        loss1=loss2
    grad = np.matmul(np.matmul(A.T,A),x_solved)-np.matmul(A.T,b_noise)

    #print(grad)
    #print((lr/num_train)* grad)
    x_solved = x_solved - lr* grad
    loss = np.mean((np.matmul(A,x_solved)-b_noise)**2)
    #print("loss")
    #print(loss)

    #print("x")
    #print(x_solved)
    #print(np.sqrt(np.mean(grad**2)))
    print("grad norm", np.linalg.norm(grad,2), "loss", loss, "x", x_solved.flatten(),"lr",lr, "iter", iter_ctr)
    #print(np.linalg.norm(grad,2))
    #print(iter_ctr)
    iter_ctr += 1

    '''
    if iter_ctr == lr_ctr*1000:
        lr = lr/10
    '''
print("Finished.")
print("x")
print(x_solved)
#print(iter_ctr)
