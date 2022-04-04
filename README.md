# Video Frame Interpolation Using Neural Networks

The task of this semester task was to research, implement and evaluate neural network models that are capable of video frame interpolation which results in frame rate upscaling. I used CNN and DCGAN.

## Relevant links

Milestone link: [here](milestone/milestone.pdf)

Report link: [here](report/report.pdf)

## How to rerun preprocessing and training

-----

I supply guide to rerun the steps in google colab environment as this was mainly the enviroment I used after realizing that training on my cpu would take ages. Guide contains links to my google drive under my faculty account. Everyone with FIT google account should have an access to files in the links.

1. Download src folder and upload it to your Google drive.
2. (Optional) Frame extraction
    1. Download videos from: <https://drive.google.com/drive/folders/1V25Z8ab8pWcSbjkmADN8Yk54SpXmaXM2?usp=sharing>
    2. Paste them to ./src/data/videos
    3. Create partitions folder in ./src/data
    4. Run preprocess.ipynb (change src path in cd command)
3. If you skipped 2nd step
   1. Link to extracted partitioned frames: <https://drive.google.com/drive/folders/1MDYdiUsl_Hp-FFp6hkUvncw-64mYScaH?usp=sharing>
   2. Create partitions folder in ./src/data/
   3. Download all partitions and paste to ./src/data/partitions/
4. Training
   1. (Here are model weights to skip training) Download weights here: <https://drive.google.com/drive/folders/1-5SaGowcW2VFqAR742h6eNbZ95ZeVXCa?usp=sharing>
   2. Create weights folder in ./src/ and paste downloaded .hdf5 files here
   3. Run train.ipynb (change src path in cd command)
5. Sample output
   1. Run Sample.ipynb (change src path in cd command)
