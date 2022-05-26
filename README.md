# Demo of my issue with Laplace diagonal

## How to obtain training data?
Download it via commands 
If kaggle not set up:\
`pip install kaggle` 
set up kaggle.json in ~/.kaggle (on linux). For more details see https://www.kaggle.com/docs/api -> Authentication

Then run:\
`kaggle competitions download -c facial-keypoints-detection`

However, this is not necessary to run. You can use the state_dict.dill from a pretrained model to go straight to Laplace step. 
