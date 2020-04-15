# ScamCanner

## Instructions for use

The project currently incomplete and the instructions will be available once it is ready.

## Using the SURF detector and classifying feature

Currently the work is to classify whether a feature is a corner or not. The SurfCorner.py is a python3 program that reads images from "ImageData" directory and does some filtering and then uses SURF detector to get Keypoints. We take top 30 keypoints if there exist and display it one by one on the image and you have to type 1 or 0 depending on whether it is a true corner(approximately). It will save it in a csv file after each image is complete.

After loading images please be patient as it takes few milliseconds to apply the filters and perfrom SURF. If you want to see the overall features for all images uncomment the big part of code and comment the rest till the end....

*Note: The images have to be .jpg for it to be able to detect in the directory.

## Testing the trained corner detector

Put the data.csv logreg.py and test.py in a single directory. Rename your image file as 'test.jpg'. Run test.py.

## Using clickSURF

Put test images in a directory ImageData. Run script. For selecting corner area-press left click at one diagonal end drag cursor to next diagonal end and release. Mark as many rectangles as needed. Press 'r' if you want to redo for the current page. Press if you want to confirm the selected portions.
