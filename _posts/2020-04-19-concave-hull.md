--- 
title: "Computing the Concave Hull of a set of points in 2D"
date: 2020-04-19
layout: "single"
permalink: /Concave-hull/
tagline: ""
mathjax: "true"
---


[Link to Repository](https://github.com/M-Lin-DM/Concave-Hulls)
See **hulls.py** and **test_ConcaveHull.py**

# Project Description




Here I have modified João Paulo Figueira's python implementation of hulls.py so that it takes ordinary 2D Cartesian coordinates as input, and returns a concave hull using the minimum possible value of k, the only parameter required by this algorithm [1]. The original code [2] was designed to work with latitude-longitude data, and uses the appropriate distance metrics. My code uses the standard euclidean distance metric and increases k by 1 until a concave hull is found which encloses all points and does not self-intersect.

![im](/images/concavehull/hull_of_dat.png)
# References

1. Original paper explaining the algorithm: Moreira, A. and Santos, M.Y., 2007, Concave Hull: A K-nearest neighbors approach for the computation of the region occupied by a set of points [https://towardsdatascience.com/the-concave-hull-c649795c0f0f](https://towardsdatascience.com/the-concave-hull-c649795c0f0f)
2. João Paulo Figueira's concave hull repository for Geospatial data [https://github.com/joaofig/uk-accidents](https://github.com/joaofig/uk-accidents)
3. João Paulo Figueira's towardsdatascience article explaining code [https://towardsdatascience.com/the-concave-hull-c649795c0f0f](https://towardsdatascience.com/the-concave-hull-c649795c0f0f)