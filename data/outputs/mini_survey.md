# Introduction
Object detection is a fundamental task in computer vision with wide applications, including self-driving cars, surveillance systems, and medical imaging. The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Traditional object detection models often rely on complex attention mechanisms, which can be computationally expensive and difficult to train.

DETR-based methods have gained significant attention in recent years due to their simplicity, efficiency, and effectiveness. This survey aims to provide an overview of the latest advancements in DETR-based object detection methods, including Deformable DETR, Conditional DETR, and DAB-DETR.

# Paper 1: DETR
The first paper introduces the DETR (DEtection TRansformer) model, which views object detection as a direct set prediction problem. DETR uses a transformer encoder-decoder architecture to predict all objects at once and is trained end-to-end with a set loss function that performs bipartite matching between predicted and ground-truth objects.

## Main Contribution
The main contribution of this work is the development of a simple and efficient object detection model that achieves comparable results to state-of-the-art detectors on challenging benchmarks. DETR demonstrates accuracy and run-time performance on par with Faster R-CNN, while being more straightforward to implement and extend to other tasks such as panoptic segmentation.

# Paper 2: Deformable DETR
The second paper proposes Deformable DETR, which replaces traditional Transformer attention modules with deformable attention modules. This approach mitigates slow convergence and high complexity issues in DETR.

## Main Contribution
Deformable DETR achieves better performance than DETR, especially on small objects, with 10 times less training epochs. The authors also propose two variants: iterative bounding box refinement and two-stage Deformable DETR.

# Paper 3: Conditional DETR
The third paper introduces a conditional cross-attention mechanism for object detection. This approach learns a spatial query from the corresponding reference point and decoder embedding, allowing it to mine spatial information while reducing dependence on content queries.

## Main Contribution
Our method simplifies attention mechanisms in object detection, reducing computational complexity and training difficulty. By learning a spatial query from reference points and decoder embeddings, we enable more efficient and effective object detection.

# Paper 4: DAB-DETR
The fourth paper proposes DAB-DETR, which uses dynamic anchor boxes for DETR (Detectron2). This approach introduces anchor boxes as queries, allowing for better positional prior with temperature tuning and size-modulated attention to account for objects of different scales.

## Main Contribution
Our method provides a deeper understanding of the roles of queries in DETR and improves the interpretability of this important submodule. We demonstrate that using dynamic anchor boxes can lead to better object detection results.

# Method Comparison

| Paper | Model | Dataset | Backbone | AP | AP50 | AP75 |
| --- | --- | --- | --- | --- | --- | --- |
| 1    | DETR-R101 | COCO | R101 | 33.0 | - | - |
| 2    | Deformable DETR | COCO 2017 test-dev set | ResNeXt-101 + DCN | 52.3 | 71.9 | 58.1 |
| 3    | DAB-DETR | COCO-Instance | DC5-R50 | 43.8 | 39.4 | 36.9 |
| 4    | DAB-DETR | COCO 2017 validation set | ResNet-50-DC5 | 45.8 | 63.1 | 44.9 |

## Main Differences
The main differences between the proposed methods are:

* DETR and Deformable DETR use traditional Transformer attention modules, while Conditional DETR introduces a conditional cross-attention mechanism.
* DAB-DETR uses dynamic anchor boxes for DETR (Detectron2), which improves the interpretability of this important submodule.

## Research Progress
The methods evolved from one paper to another by introducing new mechanisms and architectures. Deformable DETR improved upon DETR by mitigating slow convergence and high complexity issues. Conditional DETR simplified attention mechanisms in object detection, reducing computational complexity and training difficulty. DAB-DETR introduced dynamic anchor boxes for DETR (Detectron2), which led to better object detection results.

## Experimental Comparison

* AP: Deformable DETR achieved the highest AP on the COCO 2017 test-dev set with 52.3.
* AP50 and AP75: Conditional DETR performed well on these metrics, achieving 71.9 and 58.1 respectively.
* Backbone: ResNeXt-101 + DCN was used in Deformable DETR, while ResNet-50-DC5 was used in DAB-DETR.
* Dataset: COCO 2017 test-dev set was used for Deformable DETR, while COCO-Instance and COCO 2017 validation set were used for Conditional DETR and DAB-DETR respectively.

# Conclusion
This survey provides an overview of the latest advancements in DETR-based object detection methods. The proposed methods have shown promising results, with Deformable DETR achieving the highest AP on the COCO 2017 test-dev set. The introduction of dynamic anchor boxes for DETR (Detectron2) has led to better object detection results. Future research directions include exploring more practical variants of end-to-end object detectors and improving the interpretability of attention mechanisms in object detection.