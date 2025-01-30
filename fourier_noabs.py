import numpy as np
import matplotlib.pyplot as plt
import cmath

def driver():
    x = np.linspace(-np.pi,np.pi,15)
    f = lambda x: 1/(1+x**2)

    y=f(x)


    n = len(y)
    Y = np.fft.fft(y)
    ifft = np.fft.ifft(Y) # the sinusoid you're looking for

    COMPONENTS = n


    Y = np.fft.fft(y)
    np.put(Y, range(n, n), 0.0)
    ifft = np.fft.ifft(Y)
    plt.plot(x, ifft, alpha=.70)


    #plt.plot(x,y, label="Original dataset")
    plt.grid(linestyle='dashed')
    plt.legend()
    plt.show()

    arr = np.zeros(len(x))
    index =0
    for xval in x:
        arr[index] = evalA(xval,n,np.real(ifft))



    #print(ifft,alpha=0.7)
    #plt.plot(x,evalA(x,n,ifft))
    print(evalA(x,n,Y))
    plt.plot(x,arr)
    #print(np.real(ifft))
    #plt.plot(x,np.real(ifft))
    #plt.plot(x,np.imag(ifft))
    plt.show()


def evalA(z,N,iFFT):
    n=np.linspace(1,N,N)
    coeff = 1/n

    return np.sum(coeff*np.real(iFFT))




    
driver()