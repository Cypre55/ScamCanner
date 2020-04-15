# ScamCanner

## Instructions for use

The project currently incomplete and the instructions will be available once it is ready.

## Using the SURF detector and classifying feature

Currently the work is to classify whether a feature is a corner or not. The SurfCorner.py is a python3 program that reads images from "ImageData" directory and does some filtering and then uses SURF detector to get Keypoints. We take top 30 keypoints if there exist and display it one by one on the image and you have to type 1 or 0 depending on whether it is a true corner(approximately). It will save it in a csv file after each image is complete.

After loading images please be patient as it takes few milliseconds to apply the filters and perfrom SURF. If you want to see the overall features for all images uncomment the big part of code and comment the rest till the end....

*Note: The images have to be .jpg for it to be able to detect in the directory.
