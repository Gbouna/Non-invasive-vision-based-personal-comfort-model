# Non-invasive vision-based personal comfort model using thermographic images and deep learning

This is the official repository for **Non-invasive vision-based personal comfort model using thermographic images and deep learning**, our paper published in the Journal of Automation in Construction. 

# Prerequisites
Create a conda environment and install the following:

1. Tensorflow
2. Matplotlib
3. Scikit-learn 
`
# Data Preparation
Your dataset should have the following structure: 
```
- 3_TSV_base_dir/
  - participant 1/
  - participant 2/
  ...
      - Cool
      - Neutral
      - Warm
```

The same file structure should be used for 7 TSV.

# Training
Run the following script for 7 TSV training `python main.py --base_dir ./base_directory --num_classes 7`

Run the following script for 3 TSV training `python main.py --base_dir ./base_directory --num_classes 3`

