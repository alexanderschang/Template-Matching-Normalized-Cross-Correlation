# Template-Matching-Normalized-Cross-Correlation
Python implementation of template matching using normalized cross correlation formulas. 
The file contains 3 separate functions:
1. normxcorr2(template, image) computes normalized cross correlation scores between a given template and a search image, returning a matrix of normalized cross correlation (ncc) scores;

2. find_matches(template, image, thresh=None) finds the best match (of ncc scores) and returns the (x,y) coordinates of the upper left corner of the matched region in original image. When thresh is given, returns a list of (x,y) coordinates above the threshold; also returns match_image, a copy of the original search image where all matched regions are marked;

3. Driver function to load input images and matched results. 
