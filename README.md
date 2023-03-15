# Advanced-Computer-Architectures
Course material for Advanced Computer Architectures course.

## Getting started with the repo

### Clone the repository

```bash
git clone https://github.com/PARCO-LAB/Advanced-Computer-Architectures.git
```

Follow the lessons and solve the exercises by keeping your solutions local.

### Fork the repository and save the changes

If you want to save your solutions remotely, fork the main repository: 

![Alt Text](https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/fork.png)

Clone the repository forked in your GitHub profile and solve the exercises. When you’re ready to submit your changes, stage and commit your changes as the following:
```bash
git add .
git commit -m "<comment to the changes>"
git push
```

In this way, your exercise implementations are saved in your personal GitHub profile.

## Usage
In the exercise directory, create the build folder, launch the cmake command and compile the sources:
```bash
cd <lesson-folder>/<exercise-name>

# Only the first time
mkdir build
cd build
cmake ..

# Every time you have to recompile
make
./<executable_name>
```

## Benchmarking

Add the following line to the main function in the source file:
```c++
std::cout << TM_host.duration() << ";" << TM_device.duration() << ";" << TM_host.duration() / TM_device.duration() << std::endl;
```

Then launch the benchmarking script in the build folder:
```bash
../../../utils/benchmark.sh <executable_name>
```

If you are using a Jetson board, make sure to set the power profile to max, typing:
```bash
sudo nvpmodel -m<n> # (The number may change, depending on the device, for TX2 <n>=0)
sudo jetson_clocks
```

## Hardware details (only for NVIDIA Jetson)
To have a complete overview of the Jetson hardware status, type:
```bash
sudo jtop
```

If it isn't already installed, type:
```bash
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jetson_stats.service
```