## 00. 1DStencil (Shared Memory)

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l2_00.jpg" width="500" height=auto> 

### Performance reference table (N = 100 000 000):
CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
13640 ms   | 92 ms    | 149x     | Nvidia Jetson TX2  | No Shared | -
13652 ms   | 127 ms    | 106x     | Nvidia Jetson TX2  | Shared Mem | -


## 01. MatrixMultiplication (Shared Memory)

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l2_01.jpg" width="500" height=auto> 

### Performance reference table (N = 1024):

CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
52100 ms   | 84 ms    | 614x     | Nvidia Jetson TX2  | No Shared | -
57194 ms   | 25 ms    | 2198x     | Nvidia Jetson TX2  | Shared Mem | -


## 02. MatrixTranspose (Shared Memory)

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l2_02.jpg" width="500" height=auto> 

### Performance reference table (N = 8192):
CPU time   | GPU time | Speedup  | Device             | Mode       |Author
-----------| -------- | -------- | ------------------ | ---------- |------
5474 ms    | 65 ms    | 82x      | Nvidia Jetson TX2  | No Shared  | -
5465 ms    | 45 ms    | 117x     | Nvidia Jetson TX2  | Shared Mem | -
