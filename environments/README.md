# Setting up train & test environments
These are train & test environments where are with fixed perturbed dynamics.

I changed one or more factors in the xml file of each perturbed environment:
* For gravity perturbation, I changed "gravity".
* For mass perturbation, I changed "density" or "settotalmass".
* Environment dynamics factors in the xml files were changed from the nominal value of the factors.

If you have a problem while you run experiments, please follow the below instead.\
Go to the directory `home/[user name]/anaconda3/envs/[your virtual env]/lib/python3.7/site-packages/gym/` (If you can't find the `gym` folder in the directory, please check the package version),\
then replace all the existing files/folders with our files/folders
