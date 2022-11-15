# Overview of Progress

Overview of the multiple implementen algorithm. We have the standard Horn Schunck implemented on gray scale image. 
The Horn Schunck with multiple channels for color images. Finally, we have the Horn Schunck with Multiple Channels, Multi Resolution and Iterative Refinement.

| *Image*  | *HS Gray*  | *HS Color* | *HS MC+MR+IR* |
| ------------- | ------------- | ------------- |-------------  | 
| Dimetrodon  | ![](/results/result-other-grey-HS/Dimetrodon.png)  | ![alt text](/results/result-other-color-HS/Dimetrodon.png)  | ![alt text](/results/results-other-color-MR-IR-HS/Dimetrodon.png)|
| Grove2  | ![alt text](/results/result-other-grey-HS/Grove2.png)  | ![alt text](/results/result-other-color-HS/Grove2.png)  |![alt text](/results/results-other-color-MR-IR-HS/Grove2.png) |
| Grove3  | ![alt text](/results/result-other-grey-HS/Grove3.png)  | ![alt text](/results/result-other-color-HS/Grove3.png)  | ![alt text](/results/results-other-color-MR-IR-HS/Grove3.png) |
| Hydrangea  | ![alt text](/results/result-other-grey-HS/Hydrangea.png)  | ![alt text](/results/result-other-color-HS/Hydrangea.png)  |![alt text](/results/results-other-color-MR-IR-HS/Hydrangea.png)  |
| RubberWhale  | ![alt text](/results/result-other-grey-HS/RubberWhale.png)  | ![alt text](/results/result-other-color-HS/RubberWhale.png)  | ![alt text](/results/results-other-color-MR-IR-HS/RubberWhale.png)  |
| Urban2 | ![alt text](/results/result-other-grey-HS/Urban2.png)  | ![alt text](/results/result-other-color-HS/Urban2.png)  |![alt text](/results/results-other-color-MR-IR-HS/Urban2.png) |
| Urban3 | ![alt text](/results/result-other-grey-HS/Urban3.png)  | ![alt text](/results/result-other-color-HS/Urban3.png)  |![alt text](/results/results-other-color-MR-IR-HS/Urban3.png) |
| Venus | ![alt text](/results/result-other-grey-HS/Venus.png)  | ![alt text](/results/result-other-color-HS/Venus.png)  |![alt text](/results/results-other-color-MR-IR-HS/Venus.png) |

# Comparing best results to the true flow
| *Image*  | *HS MC+MR+IR*  | *True flow* |
| ------------- | ------------- | ------------- |
| Dimetrodon  |  ![alt text](/results/results-other-color-MR-IR-HS/Dimetrodon.png)|![](/ground_truth_flow/Dimetrodon/Dimetrodon.png)|
| Grove2  |  ![alt text](/results/results-other-color-MR-IR-HS/Grove2.png) |![](/ground_truth_flow/Grove2/Grove2.png)|
| Grove3  | ![alt text](/results/results-other-color-MR-IR-HS/Grove3.png) |![](/ground_truth_flow/Grove3/Grove3.png)|
| Hydrangea  | ![alt text](/results/results-other-color-MR-IR-HS/Hydrangea.png)  |![](/ground_truth_flow/Hydrangea/Hydrangea.png)|
| RubberWhale  | ![alt text](/results/results-other-color-MR-IR-HS/RubberWhale.png)  |![](/ground_truth_flow/RubberWhale/RubberWhale.png)|
| Urban2 | ![alt text](/results/results-other-color-MR-IR-HS/Urban2.png) |![](/ground_truth_flow/Urban2/Urban2.png)|
| Urban3 | ![alt text](/results/results-other-color-MR-IR-HS/Urban3.png) |![](/ground_truth_flow/Urban3/Urban3.png)|
| Venus | ![alt text](/results/results-other-color-MR-IR-HS/Venus.png) |![](/ground_truth_flow/Venus/Venus.png)|


# Comparing our Average End-Point-Error statistics to the statistics of the professor
| *Image*  | *HS MC+MR+IR*  | *Professor* |
| ------------- | ------------- | ------------- |
| Dimetrodon | 2.11 (1.52)| 0.454  |
| Grove2 | 3.21 (1.71)| 0.290  |
| Grove3 | 3.9 (2.73)| 0.854 |
| Rubber Whale | 1.25 (1.57) | 0.432 |
| Hydrangea | 3.74 (1.95)| 0.495  |
| Urban2 | 8.52 (8.59)| 0.330  |
| Urban3 | 7.5 (5.53) | 0.800  |

# Comparing our Average Angular Error statistics to the statistics of the professor

| *Image*  | *HS MC+MR+IR*  | *Professor* |
| ------------- | ------------- | ------------- |
| Dimetrodon | 46 (26)| 10.02 (17.03)  |
| Grove2 | 30 (27)| 3.9 (7.96)  |
| Grove3 | 35 (34)| 7.29 (14.91) |
| Hydrangea | 32 (28)| 6.72 (14.09)  |
| RubberWhale | 66 (18)| 10.70 (20.08) |
| Urban2 | 9 (58)| 5.57 (15.93) |
| Urban3 | 4 (54) | 12.98 (27.58)  |