# ScamCanner
<<<<<<< HEAD
## How to run?
To run the ScamCanner make a local directory called "ImageData" and add the images whose documented version you would like to see. If it is unsucessful for a particular image then it will print no document found in the console.

## Made by
Agni Purani, Neha Dalmia, Rishab Agarwal, Rohit Raj, Rohit Sutradhar, Satwik Chappidi

Helped by: Archit Rungta, Balaji Udayagiri, Debjoy Saha, Raja Raghav, Yash Soni

## Modules required in python3
astroid==2.3.3
cycler==0.10.0
decorator==4.4.2
imageio==2.8.0
imbalanced-learn==0.6.2
imblearn==0.0
imutils==0.5.3
isort==4.3.21
joblib==0.14.1
kiwisolver==1.2.0
lazy-object-proxy==1.4.3
matplotlib==3.2.1
mccabe==0.6.1
networkx==2.4
numpy==1.18.2
pandas==1.0.3
Pillow==7.1.1
pylint==2.4.4
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2019.3
PyWavelets==1.1.1
scikit-image==0.16.2
scikit-learn==0.22.2.post1
scipy==1.4.1
seaborn==0.10.0
six==1.14.0
typed-ast==1.4.1
wrapt==1.11.2
=======

## Instructions for use

<<<<<<< HEAD
The project currently contains 2 directories "Surf_test" and "Testing_annotation using surf". The first one contains a python program to read images from a directory name "ImageData" inside it(Make yourself) so it will generate a csv file based on the inputs taken from the user for each image in the directory. The second one works similarly but is for classififing corners only.
>>>>>>> log_reg_test
=======
The project currently incomplete and the instructions will be available once it is ready.

## Using the SurfCorner.py detector and classifying feature

*USE clickSURF.py for faster and better marking.*

Currently the work is to classify whether a feature is a corner or not. The SurfCorner.py is a python3 program that reads images from "ImageData" directory and does some filtering and then uses SURF detector to get Keypoints. We take top 30 keypoints if there exist and display it one by one on the image and you have to type 1 or 0 depending on whether it is a true corner(approximately). It will save it in a csv file after each image is complete.

After loading images please be patient as it takes few milliseconds to apply the filters and perfrom SURF. If you want to see the overall features for all images uncomment the big part of code and comment the rest till the end....

*Note: The images have to be .jpg for it to be able to detect in the directory.*

## Testing the trained corner detector

Put the data.csv logreg.py and test.py in a single directory. Rename your image file as 'test.jpg'. Run test.py.

## Using clickSURF

Put test images in a directory "ImageData". Run script. For selecting corner area-press left click at one diagonal end drag cursor to next diagonal end and release. Mark as many rectangles as needed. Press 'r' if you want to redo for the current page. Press 'c' if you want to confirm the selected portions and the final output will shown which ones are marked 1 and the result will be stored in "Data4Training.csv". 

The images in the folder "ImageData" **WILL BE DELETED** after its data has been added. It is advisable to **COPY** the images to the directory. This feature is added so if you encounter an error you can continue from where you left off. If you would like to remove this, then remove the last line of "clickSURF.py".
```python
		os.remove(img_path)
```

>>>>>>> SURF_corners
