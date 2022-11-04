# Setting up train & test environments
For train & test environments where are with fixed perturbed dynamics, please try the follow.

In the directory `home/anaconda3/envs/"your virtual name"/lib/python3.7/site-packages/gym/` (If you can't find the `gym` folder in the directory, please check the package version),\
I changed one or more factors in the xml file of each perturbed environment:
* For gravity perturbation, we changed "gravity" in xml files.
* For mass perturbation, we chaned "density" or "settotalmass".

***
I've been checking if uploading our `envs.zip` may cause a trouble like intellectual property conflicts.
If the zip file replace exists in this `environments` folder, just replace the original `envs` folder with the provided `envs` folder.

