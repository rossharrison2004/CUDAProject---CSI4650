CUDA Project - CSI 4650
Professor Jiannan Tian

Group members: Ross Harrison, Alyssa Darakdjian, Christian Peretz

Project explanation:

Compile: 
On Linux run: 
nvcc -03 -arch=sm_86 reduction.cu -o reduction

Use your GPU’s compute capability:
RTX 30-series → sm_86
RTX 20-series → sm_75
GTX 10-series → sm_61

Run: 
./reduction
