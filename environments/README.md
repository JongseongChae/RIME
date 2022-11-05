# Setting up train & test environments
For train & test environments where are with fixed perturbed dynamics, please try the follow.

In the directory `home/anaconda3/envs/"your virtual name"/lib/python3.7/site-packages/gym/` (If you can't find the `gym` folder in the directory, please check the package version),\
I changed one or more factors in the xml file of each perturbed environment:
* For gravity perturbation, I changed "gravity".
* For mass perturbation, I chaned "density" or "settotalmass".
* Environment dynamics factors in the xml files were changed from the nominal value of the factors.

***
I've been checking if uploading our `envs.zip` may cause a trouble like intellectual property conflicts.
If the zip file exists in this `environments` folder, just replace the original `envs` folder, which is in the directory, with the provided `envs` folder.

