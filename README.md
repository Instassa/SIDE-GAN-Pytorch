### SIDE-GAN-Pytorch code for adversarial generation of dynamic system ID trajectories.

This is the accompanying code for the paper 'Adversarial Generation of Informative Trajectories for Dynamics System Identification', published in IEEE iROS2020.

If you would like a short video explanation please follow this link: https://youtu.be/N32WzBEAIFM

### Credit:

If you intend to use any of this for a publication please cite 
> _"Adversarial Generation of Informative Trajectories for Dynamics System Identification," M. Jegorova, J. Smith, M. Mistry and T. Hospedales, 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)._
https://arxiv.org/pdf/2003.01190.pdf



### Robot, simulator, training data, and other code-specific details:

> **Prerequisites to run this code are as follows:**  
>    <code> python >=3.5.3</code>  
>    <code> pytorch >= 1.3.0</code>  
>    <code> numpy >= 1.13.3</code>  
>    <code> os</code>  
>    <code> time  </code>


We provide a small training dataset here, however, as it is a subsample of the training data used for the aforementioned publication the results are unlikely to be the same. If for whatever reason you require the full dataset, please contact m.jegorova@ed.ac.uk - I will be happy to provide this data.

Please note that the trajectories provided are for KUKA LWR IV manipulation platforms. The generation procedure is likely to generalise to other platforms, but you will have to acquire the training trajectories parameters elsewhere (in our case we used modified Fourier transform parameters, as described in the paper). 

For visualising these trajectories you would need to use [ARDL library]{https://github.com/smithjoshua001/ARDL}, please make sure you use the correct mark of a manipulator.

### To run the code

<code> python3 DCGAN_pytorch_double_best_fit_div_SR.py --dataset fresh_folder --dataroot ./data_for_pytorch_small </code>




