# Lights as Points: Learning to Look at Vehicle Substructures with Anchor-Free Object Detection
Lights as Points: Learning to Look at Vehicle Substructures with Anchor-Free Object Detection:            


## Abstract
Vehicle detection is a paramount task for safe autonomous driving, as the ego-vehicle must localize other surrounding vehicles for safe navigation. Unlike other traffic agents, vehicles have necessary substructural components such as the headlights and tail lights, which can provide important cues about a vehicle’s future trajectory. However, previous object detection methods still treat vehicles as a single entity, ignoring these safety-critical vehicle substructures. Our research addresses the detection of substructural components of vehicles in conjunction with the detection of the vehicles themselves. Emphasizing the integral detection of cars and their substructures, our objective is to establish a coherent representation of the vehicle as an entity. Inspired by the CenterNet approach for human pose estimation, our model predicts object centers and subsequently regresses to bounding boxes and key points for the object. We evaluate multiple model configurations to regress to vehicle substructures on the ApolloCar3D dataset and achieve an average precision of 0.782 for the threshold of 0.5 using the direct regression approach.

## Code Reference
This is the implementation in Python of the paper submitted to the Robotics and Automation Letters journal: Lights as Points: Learning to Look at Vehicle Substructures with Anchor-Free Object Detection. The repository was forked off the CenterNet repository that has the implementation of the paper Object as Points. The code for the paper Objects as Points used for human pose estimation has been replaced by the code for detecting cars in conjunction with their substructures, namely the headlights, tail lights, and license plates. The code that is common to both implementations, the human pose estimation and vehicle subtructure detection, has been left unchanged except for the minor changes required to make the code work.

## Main Results

### Object Keypoint Similarity Results per Substructure Per Model Without Visibility Models
All the results are $$(AP_{0.5}, AP_{0.75})$$.

| Model     |  Left Headlight | Left Tail Light|  Right Tail Light | Right Headlight | Front License Plate | Rear License Plate |
|--------------|-----------|--------------|-----------------------|
|Direct Regression Model | 0.884, 0.823 | 0.838, 0.742 | 0.833, 0.734 | 0.877, 0.828 | 0.906, 0.872 | 0.794, 0.715 |
|Double Regression Model | 0.794, 0.655 | 0.628, 0.359 | 0.641, 0.412 | 0.825, 0.732 | 0.826, 0.707 | 0.473, 0.352 |
|Substructure Center Regression Model | 0.864, 0.837 | 0.786, 0.783 | 0.809, 0.789 | 0.884, 0.883 | 0.894, 0.889 | 0.750, 0.716 |


## Train the model

### Clone the repo

You can clone the repo with the command 
~~~
git clone https://github.com/mmkeskar/Vehicle-Substructures-Detection.git
~~~

### Train the Model



