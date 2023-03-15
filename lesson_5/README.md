## 00. Vector Reduction

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l3_00.jpg" width="500" height=auto> 

Suggestion: use BLOCK_SIZE=256. Note that [@micheleboldo](https://github.com/micheleboldo)'s implementation uses BLOCK_SIZE=128.
Starting from the idea of [@micheleboldo](https://github.com/micheleboldo), the implementation of [@xaldyz](https://github.com/xaldyz) uses BLOCK_SIZE=64.

### Performance reference table (N = 16777216):
CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
73 ms   | 21 ms    | 3x      | Nvidia Jetson TX2  | Shared Divergence | -
73 ms   | 12 ms    | 5x      | Nvidia Jetson TX2  | Shared Less Divergence | -
73 ms   | 9 ms     | 7x      | Nvidia Jetson TX2  | Last Warp Opt | -
73 ms   | 5 ms     | 13x     | Nvidia Jetson TX2  | Custom | [@micheleboldo](https://github.com/micheleboldo)
73 ms   | 3 ms     | **24x**     | Nvidia Jetson TX2  | Custom | [@xaldyz](https://github.com/xaldyz)

### Performance reference table - Other Devices (N = 16777216):
CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
32.9 ms   | 3.8 ms    | 8.7x     | i7 6700K, GTX 1060 (6GB) | Shared Divergence | [@micheleboldo](https://github.com/micheleboldo)
33.4 ms   | 2.1 ms    | 3.5x     | i7 6700K, GTX 1060 (6GB)  | Shared Less Divergence | [@micheleboldo](https://github.com/micheleboldo)
33.3 ms   | 0.9 ms    | 13x     | i7 6700K, GTX 1060 (6GB)  | Custom | [@micheleboldo](https://github.com/micheleboldo)

## 01. Prefix Scan

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l3_01.jpg" width="500" height=auto> 

### Performance reference table (N = 67108864):

CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
716 ms   | 73 ms    | 9x     | Nvidia Jetson TX2  | Naive | -
722 ms   | 110 ms   | 6x     | Nvidia Jetson TX2  | Up Down Sweep | -
