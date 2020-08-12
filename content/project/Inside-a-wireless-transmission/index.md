---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Inside a Wireless Transmission"
summary: "Wireless speakers, internet signals, headset, etc., are part of our daily life. Nevertheless, understanding how it works is essential to create new gadgets. This post will focus on FSK modulation. A digital modulation technique, present in our daily lives."
authors: [Miguel Hisojo]
tags: [Signal processing, Data Science]
categories: [Signal processing]
date: 2020-08-12T16:26:58+02:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Wireless transmissions"
  focal_point: "center"
  preview_only: true

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

![wireless.png](featured.png)



# How your wireless speaker works?
## An FSK simulation


Mobile phones, internet access, and to listening music on the radio are normal activities for most of us. When any of those activities take place, we impose a data signal wave over a carrier signal wave, generally of higher frequency. This process is called modulation.

Digital Modulation provides more information capacity, high data security, quicker system availability with great quality communication.

Frequency Shift Keying (FSK) is one of the most common modulation techniques. FSK modulates a signal so it can be transmitted wirelessly. Bluetooth, uses Gaussian FSK modulation, a slightly different version of FSK. This notebook aims to provide an overview of how data transmission works over FSK.




# System description.

## Transmitter

FSK modulation uses two frequencies to represent binary values of 1 and 0, so our input data d will be converted to these frequencies. Our transmitter (tx) will simply create two distinct frequencies based on a data value, by using a voltage-controlled oscillator (VCO).

{{< figure src="tx_fsk.png" title="FSK Transmitter" numbered="true" >}}

The *transmitted signal* **s(t)** is defined as 

$$ s(t) = A \cos(2 \pi  \omega (t) ) $$

Where **A** is the signal amplitude and $\omega (t)$ is the output of the voltage-controlled oscillator (VCO) defined as: 

$$   f_c + m(t)  = \begin{cases} 
                  1  &  f_c - f_{dev} & \\\\
                  0  &  f_c + f_{dev}
                 \end{cases}
$$


where $ f_{dev}$ is the frequency deviation, so, our output signal is represented as,
 
$$
 s(t) = A \cos(2 \pi  (f_c  + m(t)) * t )
$$

# Let's code.

```python
# Import the libraries necessary for our simulation
import numpy as np
import pylab as pl
from numpy.random import sample
import scipy.signal as signal
import scipy.signal.signaltools as sigtool
pl.rcParams['figure.figsize'] = [15, 5]    # Standard size for all plots to improve visualization.
pl.rcParams['figure.dpi'] = 300
#pl.figure(figsize=(30, 10), dpi= 80)
```


```python
#SIMULATED VALUES
Fc = 1000       #carrier frequency of 1kHz
Fbit = 50       #bitrate of data
Fdev = 500      #frequency deviation
N = 64          #Size of the array of bits that we will transmit
A = 1           #tx ignal amplitude
Fs = 10000      #sampling frequency
A_n = 0.1       #noise peak amplitude
N_pbits = 10    #number of bits to print in plots

```


```python
#generate some random data for testing
d = np.random.randint(0,1+1,N)

print('d:  \n',d)


```

    d:  
     [0 0 1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 1 0 1 1 1 1 1 1 1 0 0 0 0 0 1 1 0 0 1 1
     0 1 1 1 0 1 0 1 1 1 0 1 0 0 0 1 1 1 0 1 0 1 1 1 0 0 0]


The voltage-controlled oscillator(VCO) converts the bit stream to a sine wave, with a frequency-dependent value on whether the bit is a 1 or 0. The bit stream is then extended to match the sampling frequency.


```python
t = np.arange(0,float(N)/float(Fbit),1/float(Fs), dtype=np.float)

#extend the d to account for the bitrate and convert 0/1 to frequency
m = np.zeros(0).astype(float)
zero=int(Fc+Fdev)
one=int(Fc-Fdev)
fsbit=int(Fs/Fbit)

for bit in d:
   if bit == 0:
       m=np.hstack((m,np.multiply(np.ones(int(Fs/Fbit)),Fc+Fdev)))
   else:
       m=np.hstack((m,np.multiply(np.ones(int(Fs/Fbit)),Fc-Fdev)))
print(len(m))
        
#calculate the output of the VCO
y=np.zeros(0)
y=A * np.cos(2*np.pi*np.multiply(m,t))

pl.plot(t[0:int(Fs*N_pbits/Fbit )],y[0:int(Fs*N_pbits/Fbit)])
pl.xlabel('Time (s)')
pl.ylabel('Amplitude (V)')
pl.title('Amplitude of carrier vs time')

```

    12800

{{< figure src="FSK_6_2.png" title="Amplitude of carrier vs time" numbered="true" >}}


Frequency measurements of the FSK signal are usually stated in terms of “shift” and
center frequency. The shift is the frequency difference between the mark and space
frequencies. Shifts are usually in the range of 50 to 1000 Hertz. The nominal center 
frequency is halfway between the mark and space frequencies.  


Frequency Shift Property of the Fourier Transform:

$$
\mathscr F [x(t) \cos (2\pi f_c t)] = \frac{1}{2} X (f-f_c) + \frac{1}{2} X (f+f_c)
$$


```python
#visualize data in time and frequency domain
def visualize_data(y):
    N_FFT = float(len(y))                             
    f = np.arange(0,Fs/2,Fs/N_FFT) 
    w = np.hanning(len(y))                           
    y_f = np.fft.fft(np.multiply(y,w))
    y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]/N_FFT))
    return y_f,w ,f,N_FFT
```


```python
y_f,w,f,N_FFT=visualize_data(y)
```


```python
pl.plot(t[0:int(Fs*N_pbits/Fbit)],m[0:int(Fs*N_pbits/Fbit)])
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Original VCO output versus time')
```



{{< figure src="FSK_10_1.png" title="Original VCO output versus time" numbered="true" >}}


```python
pl.plot(f[0:int((Fc+Fdev*2)*N_FFT/Fs)],y_f[0:int((Fc+Fdev*2)*N_FFT/Fs)])
pl.xlabel('Frequency (Hz)')
pl.ylabel('Amplitude (dB)')
pl.title('Spectrum')
```




{{< figure src="FSK_11_1.png" title="Signal spectrum" numbered="true" >}}



## The channel

The signal acquires noise when passing through the channel. The thermal noise $n(t)$ is assumed to be complex additive white Gaussian (AWGN) with zero mean and power spectral density $\frac{N_0}{2}$.

The received signal can be expressed as:
$$
r(t)= s(t)+n(t)
$$

so,

$$
y(t)= A \cos(2 \pi  (f_c  + m(t)) * t ) +n(t)
$$



If the noise becomes too strong, or if the amplitude of the signal becomes too weak, the data cannot be demodulated.

In our simulation, we will set the noise amplitude of 0.1 which creates a signal-to-noise ratio of $~14dB$. Notice the noise in the time domain corrupting the signal in the following plots.



```python
#create some noise
noise = (np.random.randn(len(y))+1)*A_n
snr = 10*np.log10(np.mean(np.square(y)) / np.mean(np.square(noise)))
print("SNR = %fdB" % snr)
y=np.add(y,noise)


```
```python
    SNR = 13.961869dB
```


```python
y_f,w,f,N_FFT=visualize_data(y)

pl.plot(t[0:int(Fs*N_pbits/Fbit )],y[0:int(Fs*N_pbits/Fbit)])
pl.xlabel('Time (s)')
pl.ylabel('Amplitude (V)')
pl.title('Amplitude of carrier vs time')
print(int(Fs*N_pbits/Fbit ))
print(int(Fs*N_pbits/Fbit))

```
```python
    2000
    2000

```

{{< figure src="FSK_15_1.png" title="" numbered="true" >}}



```python
pl.plot(f[0:int((Fc+Fdev*2)*N_FFT/Fs)],y_f[0:int((Fc+Fdev*2)*N_FFT/Fs)])
pl.xlabel('Frequency (Hz)')
pl.ylabel('Amplitude (dB)')
##pl.title('Spectrum of the received signal')
```


{{< figure src="FSK_16_1.png" title="Amplitude (dB)" numbered="true" >}}


## The receiver


{{< figure src="rx_fsk.png" title="A receiver architecture" numbered="true" >}}

Our receiver will take concepts from *analog FM*, which can be demodulated by converting the **frequency changes to amplitude changes.**
Typically, FSK demodulation utilizes an analog differentiator in order to separate the data signal from the carrier frequency, followed by an envelope detector. This procedure is simple and low-power consuming.

$$
\frac{dy}{dt} = -A 2\pi \left ( f + m(t) + t * \frac{d m(t)}{dt} \right) \sin (2 \pi (fc + m(t)) * t )+ \frac{d n(t)}{dt}
$$


Combining the amplitudes in one term and taking into account that the term $\frac{d m(t)}{dt} = 0$ due to the constant phase change we have:

$$
\frac{dy}{dt} = A 2\pi \left ( f + m(t)  \right) \sin (2 \pi (fc + m(t)) * t )+ \frac{d n(t)}{dt}
$$

In our simulation, we can ignore the phase shift, since it is constant. Nevertheless, in real life, a clock recovery or bit synchronization scheme must be applied. The differentiator is simple and is just a discrete differentiation function, it can be implemented as:


```python
y_diff = np.diff(y,1)
print(y)
```

```python
    [ 1.15418648  0.56043178 -0.2061984  ... -0.76277215 -0.0358485
      0.64900518]
```


The envelope detector separates the high-frequency carrier from the low-frequency digital data modulated onto the amplitude, this is done by using a Hilbert transform. 
After the envelope detection, the signal is low-pass filtered using a 100 tap FIR filter with a cutoff frequency of $2*bitrate$.


```python
######################################################
# Envelope detector and low-pass filter
######################################################
y_env = np.abs(sigtool.hilbert(y_diff))
h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
y_low_filt=signal.lfilter( h, 1.0, y_env)

#data after adding noise
N_FFT = float(len(y_low_filt))
f = np.arange(0,Fs/2,Fs/N_FFT)
w = np.hanning(len(y_low_filt))
y_f = np.fft.fft(np.multiply(y_low_filt,w))
y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]/N_FFT))


pl.plot(t[0:int(Fs*N_pbits/Fbit)],m[0:int(Fs*N_pbits/Fbit)])
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Original VCO output vs. time')
``` 
{{< figure src="FSK_20_1.png" title="Original VCO output vs. time" numbered="true" >}}


```python
pl.plot(t[0:int(Fs*N_pbits/Fbit)],np.abs(y[0:int(Fs*N_pbits/Fbit)]))
pl.plot(t[0:int(Fs*N_pbits/Fbit)],y_low_filt[0:int(Fs*N_pbits/Fbit)],'r',linewidth=3.0)
pl.xlabel('Time (s)')
pl.ylabel('Amplitude (V)')
pl.title('Filter signal and unfiltered signal vs. time')
```

{{< figure src="/FSK_21_1.png" title="Filter signal and unfiltered signal vs. time" numbered="true" >}}



```python
pl.plot(f[0:int((Fc+Fdev*2)*N_FFT/Fs)],y_f[0:int((Fc+Fdev*2)*N_FFT/Fs)])
pl.xlabel('Frequency (Hz)')
pl.ylabel('Amplitude (dB)')
pl.title('Spectrum')
```



{{< figure src="FSK_22_1.png" title="Sectrum" numbered="true" >}}

The slicer takes the mean of the entire filtered signal, and uses this as a decision threshold, to decide if a bit is a 1 or a 0. 
The decision is done at the center of the bit period. The result is saved to an array and compared to the original, to find bit errors. 
The bit error percentage is printed to the console.


```python
#calculate the mean of the signal
mean = np.mean(y_low_filt)
#if the mean of the bit period is higher than the mean, the data is a 0
sampled_signal = y_low_filt[int(Fs/Fbit/2):len(y_low_filt):int(Fs/Fbit)]

print(sampled_signal)
pl.plot(sampled_signal)
```

    [0.94154175 0.92515726 0.35223853 0.33979885 0.35294071 0.36287615
     0.34675628 0.90095976 0.35453665 0.92090498 0.34720447 0.91611558
     0.94528336 0.34417705 0.90435061 0.91620578 0.34844842 0.3388747
     0.94404786 0.34364565 0.35304868 0.356202   0.35102639 0.34859071
     0.33627503 0.3383037  0.93701699 0.92527765 0.91127889 0.92546922
     0.90017475 0.34158569 0.34268534 0.92516404 0.91248606 0.33776427
     0.3454749  0.93246055 0.34503893 0.34917072 0.35666379 0.90326687
     0.35272848 0.92576273 0.35310379 0.34613621 0.38067857 0.88376517
     0.35971779 0.88856201 0.93363287 0.89404895 0.34008713 0.34691357
     0.33626758 0.90701975 0.33288698 0.91293492 0.34244479 0.36333365
     0.33606437 0.95159041 0.922426   0.93333146]





{{< figure src="FSK_24_2.png" title="Received sequence." numbered="true" >}}


```python
rx_data=[]
[rx_data.append(0) if bit > mean else rx_data.append(1) for bit in sampled_signal]
        
bit_error=0
for i in range(0,len(d)):
    if rx_data[i] != d[i]:
        bit_error+=1
               
print ("bit errors = %d" % bit_error)
print ("bit error percent = %4.2f%%" % (float(bit_error)/float(N)*100) )
```

    bit errors = 0
    bit error percent = 0.00%



## Conclusion

FSK is a digital modulation technique to increase the frequency characteristics of the input binary signal. By FSK modulation technique we can achieve error-free communication in a few digital applications. 
Nevertheless, FSK has finite data rate and consumes more bandwidth can be overcome by the QAM, which is known as quadrature amplitude modulation. It is the combination of amplitude modulation and phase modulation.