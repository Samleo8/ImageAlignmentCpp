# Baker-Matthews Image Alignment in C++, with speed and accuracy optimisations

This C++ application aims to implement the Baker-Matthews inverse compositional image alignment algorithm, with a robust M-estimator to add robustness against lighting conditions.

This C++ implementation is an adaption of Assignment 7 of Carnegie Mellon University's [http://16-385 Computer Vision course](http://16385.courses.cs.cmu.edu/fall2020) in Fall 2020. It is a great course: I advise CMU students to take it if possible. Lecture slides from which this implementation is derived can be  found publicly on the [course website](http://16385.courses.cs.cmu.edu/fall2020/lecture/track)

## Running the Application

0. Clone the repository

```bash
git clone https://github.com/Samleo8/ImageAlignmentCpp
```

1. Install dependencies

  See [below](#dependencies).

2. Build with `cmake`

```bash
cd build
cmake ..
```

3. Make and run the application
```bash
make .
./TestKLT
```

## Dependencies

[Eigen (min 3.3.7)](http://eigen.tuxfamily.org/index.php?title=Main_Page)

[OpenCV (suggested 4.5.*)](https://opencv.org/releases/)
