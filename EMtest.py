#coding:utf-8

#参考李航老师写的统计学习方法的第9章EM的第一个例子，三硬币模型
#假设有三枚硬币，A，B，C，每一枚扔出正面的概率分别是pi，p，q
#观察值的生成过程如下：
#先扔硬币A，如果是正面，就扔硬币B，如果是反面就扔硬币C
#观察到的变量为Y，即最后观察到的硬币正反面的序列
#其中隐变量为A的观察值z，不知道这个值是多少
import numpy as np


def generateData(pi=0.5,p=0.5,q=0.5,n=100):
    z=np.random.binomial(1,pi,n)
    y=[]
    for i in z:
        if i<0.5:
            y.append(np.random.binomial(1,p))
        else:
            y.append(np.random.binomial(1, q))
    y=np.array(y,dtype=float)
    z = np.array(z, dtype=float)
    return y,z

nsamples=1000000
y,z=generateData(pi=0.2,p=0.9,q=0.5, n=nsamples)
print 'y=',
print y
print 'z=',
print z

pi=0.2
p=0.9
q=0.5

# pi=np.random.uniform(0,1)
# p=np.random.uniform(0,1)
# q=np.random.uniform(0,1)


iterations=10
for it in range(iterations):
    #E步：求z的期望E_z[P(Z|Y,p,q)]
    for i in xrange(nsamples):
        z[i]=pi*np.power(p,y[i])*np.power(1-p,1-y[i])/( pi*np.power(p,y[i])*np.power(1-p,1-y[i])+(1-pi)*np.power(q,y[i])*np.power(1-q,1-y[i]) )
    #print z
    #M步：最大化P（Y，Z|pi，p，q）

    pi=sum(z)/nsamples
    p=sum((z*y))/sum(z)
    q=(sum(y)-sum(z*y))/(nsamples-sum(z))
    print pi,
    print p,
    print q
