## 01. MatrixMultiplication (OpenMP)

### Performance reference table (N = 1024):

CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
57194 ms   | 25 ms    | 2198x     | Nvidia Jetson TX2  | Shared Mem | -
52100 ms   | 84 ms    | 614x     | Nvidia Jetson TX2  | No Shared | -
8174 ms   |  -    | 3x     | Nvidia Jetson TX2  | OpenMP (-O3) | -
9793 ms   |  -    | 3x     | Nvidia Jetson TX2  | OpenMP (-O0) | -

## 02. Factorial

### Performance reference table (N = 268435456):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
1845 ms   | 454 ms    | 3x     | Nvidia Jetson TX2  | OpenMP | -


## 03. Find

Find two (given) consecutive numbers in an array.

### Performance reference table (N = 67108864):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
178 ms   | 61 ms    | 2x     | Nvidia Jetson TX2  | OpenMP | -

## 04. RC4 Chiper

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l5_04.jpg" width="500" height=auto> 

### Performance reference table (N = 256):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
72146 ms   | 63231 ms    | 1.13x     | Nvidia Jetson TX2  | OpenMP | -
