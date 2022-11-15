# Overview of Progress

Overview of the multiple implementen algorithm. We have the standard Lukas Kanade implemented on gray scale image. 
The Lukas Kanade with multiple channels for color images. Finally, we have the Lukas Kanade with Multiple Channels, Multi Resolution and Iterative Refinement.

| *Image*  | *LK Gray*  | *LK Color* | *LK MC+MR+IR* |
| ------------- | ------------- | ------------- |-------------  | 
| Dimetrodon  | ![](/results/result-other-grey-LK/Dimetrodon/Dimetrodon.png)  | ![alt text](/results/result-other-color-LK/Dimetrodon/Dimetrodon-colorwheel.png)  | ![alt text](/results/results-other-color-MR-IR/Dimetrodon/Dimetrodon-MR-IR.png)|
| Grove2  | ![alt text](/results/result-other-grey-LK/Grove2/Grove2.png)  | ![alt text](/results/result-other-color-LK/Grove2/Grove2-colorwheel.png)  |![alt text](/results/results-other-color-MR-IR/Grove2/Grove2-MR+IR.png) |
| Grove3  | ![alt text](/results/result-other-grey-LK/Grove3/Grove3.png)  | ![alt text](/results/result-other-color-LK/Grove3/Grove3-colorwheel.png)  | ![alt text](/results/results-other-color-MR-IR/Grove3/Grove3-MR+IR.png) |
| Hydrangea  | ![alt text](/results/result-other-grey-LK/Hydrangea/Hydrangea.png)  | ![alt text](/results/result-other-color-LK/Hydrangea/Hydrangea-colorwheel.png)  |![alt text](/results/results-other-color-MR-IR/Hydrangea/Hydrangea-MR+IR.png)  |
| RubberWhale  | ![alt text](/results/result-other-grey-LK/RubberWhale/RubberWhale.png)  | ![alt text](/results/result-other-color-LK/RubberWhale/RubberWhale-colorwheel.png)  | ![alt text](/results/results-other-color-MR-IR/RubberWhale/RubberWhale-MR+IR.png)  |
| Urban2 | ![alt text](/results/result-other-grey-LK/Urban2/Urban2.png)  | ![alt text](/results/result-other-color-LK/Urban2/Urban2-colorwheel.png)  |![alt text](/results/results-other-color-MR-IR/Urban2/Urban2-MR+IR.png) |
| Urban3 | ![alt text](/results/result-other-grey-LK/Urban3/Urban3.png)  | ![alt text](/results/result-other-color-LK/Urban3/Urban3-colorwheel.png)  |![alt text](/results/results-other-color-MR-IR/Urban3/Urban3-MR+IR.png) |
| Venus | ![alt text](/results/result-other-grey-LK/Venus/Venus.png)  | ![alt text](/results/result-other-color-LK/Venus/Venus-colorwheel.png)  |![alt text](/results/results-other-color-MR-IR/Venus/Venus-MR+IR.png) |

# Comparing best results to the true flow
| *Image*  | *LK MC+MR+IR*  | *True flow* |
| ------------- | ------------- | ------------- |
| Dimetrodon  |  ![alt text](/results/results-other-color-MR-IR/Dimetrodon/Dimetrodon-MR-IR.png)|![](/ground_truth_flow/Dimetrodon/Dimetrodon.png)|
| Grove2  |  ![alt text](/results/results-other-color-MR-IR/Grove2/Grove2-MR+IR.png) |![](/ground_truth_flow/Grove2/Grove2.png)|
| Grove3  | ![alt text](/results/results-other-color-MR-IR/Grove3/Grove3-MR+IR.png) |![](/ground_truth_flow/Grove3/Grove3.png)|
| Hydrangea  | ![alt text](/results/results-other-color-MR-IR/Hydrangea/Hydrangea-MR+IR.png)  |![](/ground_truth_flow/Hydrangea/Hydrangea.png)|
| RubberWhale  | ![alt text](/results/results-other-color-MR-IR/RubberWhale/RubberWhale-MR+IR.png)  |![](/ground_truth_flow/RubberWhale/RubberWhale.png)|
| Urban2 | ![alt text](/results/results-other-color-MR-IR/Urban2/Urban2-MR+IR.png) |![](/ground_truth_flow/Urban2/Urban2.png)|
| Urban3 | ![alt text](/results/results-other-color-MR-IR/Urban3/Urban3-MR+IR.png) |![](/ground_truth_flow/Urban3/Urban3.png)|
| Venus | ![alt text](/results/results-other-color-MR-IR/Venus/Venus-MR+IR.png) |![](/ground_truth_flow/Venus/Venus.png)|


# Comparing our Average End-Point-Error statistics to the statistics of the professor
| *Image*  | *LK MC+MR+IR*  | *Professor* |
| ------------- | ------------- | ------------- |
| Dimetrodon | 1.8 (0.85)| 0.392  |
| Grove2 | 2.58 (1.01)| 0.308  |
| Grove3 | 3.36 (2.37)| 0.988 |
| Hydrangea | 2.2 (1.46)| 0.345  |
| Urban2 | 8.7 (9.59)| 0.468  |
| Urban3 | 7.65 (6.71) | 0.572  |


# Conclusions

We conclude that using the informating within the color channels is informative to calculate the optical flow. 
The biggest difference we found is using multi resolution and iterative refinement. We believe that Lucas Kanade made a big jump in performance using MR and IR due to Lukas-Kanade being a local approach.

In regular optical flow method, we assume the following:
a) brightness constancy
b) small motion
c) Spatial coherence

We conclude that using the information from within the color channels is informative to calculate the optical flow.

Now, if the object were to move a larger distance then the traditional
optical flow method would work bad. This is why, we use gaussian pyramids
(coarse-to-fine) method to apply optical flow. Implementing the MR + IR algorithm gave us the biggest boost in performance due to Lukas-Kanade being a local approach.
