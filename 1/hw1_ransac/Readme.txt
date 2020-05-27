This Porject will find a lot of issues due to the certain way i started researching in the direction of the project.
I started with having less knowledge about the mathematical solutions used in during such coding and had to go through a lot of similar projects coded in python or open CV to learn better.

My major goals were getting the line drawn on the image.

I still cant figure out how to draw exact edges and i am working on doing the same. 

During the course of the project I took reference from a few similar projects and took ideas from the same.
https://github.com/joekeo/RANSAC/blob/master/main.cpp(Inspiration)
https://github.com/aerolalit/RANSAC-Algorithm
https://github.com/snavely/shapecontext/blob/master/lib/imagelib/ransac.c
https://github.com/brunokeymolen/canny/blob/master/canny.cpp
https://github.com/MalcolmMcLean/binaryimagelibrary/blob/master/canny.c

https://itom.bitbucket.io/latest/docs/07_plugins/development/openCVMat.html

https://docs.opencv.org/2.4/modules/core/doc/old_basic_structures.html?highlight=cvmat

https://docs.opencv.org/2.4/modules/core/doc/old_basic_structures.html?highlight=cvmat

Though not working in the best, it is a try and I plan to work on it more to get expected outputs.

Code Flow

- File main.c
  -> Run the main flow of program
     + Loop on all images; for each image:
       + Detect edge (by function canny())
       + On edge image:
         - (1) Apply ransac to get fitted line and inliers set
         - (2) Apply ransac to same image at (1), EXCEPT inliers set at (1)
         - (3) Repeat (1) and (2) until get enough line
       + Draw lines at (3) on original image
       + Save image that contain lines

- File ./utils/ransac.c:
  - Purpose: Do ransac fitting (function ransac_fitting())

- File ./utils/data_types.c:
  - Purpose: 
    + Implement vector type and its operations (similar C++ vector)
    + Declare data types for images (Mat*) and points (TPoint, pt*)

- File ./utils/image_tools.c:
  - Purpose: Implement image loading, cloning, saving; image processing operations (gaussian_blur, convolution, non_max_suppress, edge_tracing, hysteresis, canny, etc); line drawing; image converting (gray to rgb)



