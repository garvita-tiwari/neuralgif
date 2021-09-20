


### NeuralGIF
 
Code for  Neural-GIF: Neural Generalized Implicit Functions for Animating People in Clothing(ICCV21)
<img src="https://user-images.githubusercontent.com/25167952/132225169-99bcf89a-2916-4c6b-b630-6b44cda1c608.png" alt="" width="48%"/>
<img src="https://user-images.githubusercontent.com/25167952/132225184-d17f1fca-68d9-4b7d-85bb-5f1c4d66666e.png" alt="" width="48%"/>

We present Neural Generalized Implicit Functions (Neural-GIF), to animate people in clothing as a function of body pose. Neural-GIF learns directly from scans, models complex clothing and produces pose-dependent details for realistic animation. We show for four different characters the query input pose on the left (illustrated with a skeleton) and our output animation on the right.

### Dataset and Pretrained models
    [Data and pretrained models](https://nextcloud.mpi-klsb.mpg.de/index.php/s/FweAP5Js58Q9tsq)
    
###Installation
    1. Install kaolin: https://github.com/NVIDIAGameWorks/kaolin
    2. conda env create -f neuralgif.yml
    3. conda activate neuralgif

### Training NeuralGIF
     1. Edit configs/*yaml with correct path
            a. data/data_dir:
            b. data/split_file: <path to train/test split file> (see example in dataset folder)
            c. experiment/root_dir: training dir
            d. experiment/exp_name: <exp_name>
     2 . python trainer_shape.py --config=<path to config file>

### Generating meshes from NeuralGIF
    1. python generator.py --config=<path to config file>



### Citation:
    @inproceedings{tiwari21neuralgif,
      title = {Neural-GIF: Neural Generalized Implicit Functions for Animating People in Clothing},
      author = {Tiwari, Garvita and Sarafianos, Nikolaos and Tung, Tony and Pons-Moll, Gerard},
      booktitle = {International Conference on Computer Vision ({ICCV})},
      month = {October},
      year = {2021},
      }

