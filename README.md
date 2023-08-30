# Star Based Navigation

The current Navigation system uses a Global Positioning System (GPS) which operates based on satellites orbiting the Earth. It consists of 31 well-placed satellites that allow users with sensors and receivers to pinpoint the exact location when they are within the line of sight of at least three of those orbiting satellites.

This new method uses the position of star constellations relative to the object on Earth to find the latitude and longitude of the object. The idea is to use the well-labeled and detailed Sky Servey catalogs to match the images captured from a location on Earth with these catalogs to find latitude and longitude. 

Apps like Stellarium can tell us the exact location (right ascension and declination angle) of the constellations in the sky if we provide our latitude and longitude. This method is kind of the opposite to it, where we take images of the sky and match those images with the star catalogs to find the position of the stars (right ascension and declination angle) and then, find the latitude and longitude on Earth.

## Methodology

* For testing purposes a large image with fake dot patterns to simulate star constellations is generated.

  ![gussian_conv](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/153cbbb7-303a-4cee-8e46-50ec8a11c403)

* Small grids are separated from the large image, which will be used as a template to match the large image.

  ![s64_(184,349)](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/e93e6a7a-d9b3-4bc0-be01-e9a2c582aec8)

Methods which are used to match this template with the large image are discussed below:

### Approach 1

A Dense neural network is trained to find the coordinates of the template image in the large image via supervised learning. The model performed average with most of the templates having errors near 0.

![Figure_1](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/874cdcbb-02a8-4e45-bbc1-41d444423c45)

### Approach 2

Normalized Cross-Correlation was used for template matching. skimage.feature.match_template is used.

![patch_generation](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/c2025b80-a194-4f67-8a97-5b898c2f658a)
![template_matching](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/c5a2da3b-d521-4d01-8626-35ac3ff25bf2)

This approach performed amazing as long as the relative angle between the template and the large image was within 5 degrees But it was unusable for angles more than 5 degrees.

### Approach 3

Opencv's Feature Matching + Homography is used to match the template.

* It will find the keypoints and descriptors with Scale-Invariant Feature Transform (SIFT) for both the large image and the template.

    ![descriptors](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/10419e5d-642d-4039-a8b5-650b616392d3)

* Then the descriptors in the template are matched with the descriptors in the large image.
  
  ![matched](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/7d3766b0-0be2-4515-bc92-e8db013d8f3f)

This approach is only good if there are a lot of stars in the template image, which is not practical.

### Approach 4

The large image is divided into grids.

![grid](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/d6ba16c4-9922-498d-bc9c-19ba6015866b)

* Then 5 brightest points are taken in each grid and their cumulative distance and angles are calculated. The same is done for templates.

  ![patch_grid](https://github.com/JaskiratSudan/Star_based_navigation/assets/68187330/68aa0269-b5ca-454f-9773-3f2f76da0db9)

* In order for any template to match the large image these angles and cumulative distances should either completely or partially match.

## Result 

The combination of Approach 4 (Grid approach) along Approach 2 (Normalized Cross-Correlation) can be used to develop a scale and rotation Invariant model for template matching.

## Supervisor
Dr. Narendra Nath Patra, Indian Institute of Technology Indore

## Authors

- [@JaskiratSudan](https://github.com/JaskiratSudan)
