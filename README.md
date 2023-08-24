# Tree Counting, Crown Segmentation and Height Prediction at Scale

This repo contains code for the paper [Deep learning enables image-based tree counting, crown segmentation and height prediction at national scale](https://doi.org/10.1093/pnasnexus/pgad076. 'link to paper')

## Trained models ready for deployment

### Download models: :crossed_fingers:

https://drive.google.com/file/d/1ZNibrh6pa4-cjXLawua6L96fOKS3uwbn/view?usp=sharing

Model names:

trees - date - time - optimizer - lr - input bnads - input shape - no. training tiles - segmentation loss - density loss - boundary weight (for separate individual trees) - model . h5 


### Working with Google Earth Engine :muscle:

https://github.com/google/earthengine-community/blob/master/guides/linked/Earth_Engine_TensorFlow_tree_counting_model.ipynb

### Tips for direct depolyment :sunglasses:	

- Standardize input image patches channel-wise -> ((image-mean)/std)

- For large image, predict with overlap (stride < patch width), take max prediction where overlap

- Upsample / Downsample to match the trained input resolution (20 cm)

- Finetune with small local annotation data


## Finetune / Train using local annotation data

### Prepare your own training data :see_no_evil:


![Figure 0](figures/tree_annotation.png)

Figure 0: Preparing your own tree crown annotation dataset. Delineate tree crowns inclusively within each selected annotating area.

#### Finetuning typically requires a small local annotation dataset (< 5 k tree crowns)

Check details in paper..



## Code structure:


### Preprocessing for tree crown segmentation and count

```
python main0_preprocessing.py
```

--- :bookmark: set configs ---

config/Preprocessing.py

-------------------------------------------------------------------------------------------------------

### Train 1st model: Tree crown segmentation & density counting:

```
python main1_multitask_counting_segmentation.py
```

--- :bookmark: set configs ---

case1: same input spatial resolution: use config/UNetTraining.py

case2: inputs of different spatial resolution (only support 1/2 coarser resolution for now): use config/UNetTraining_multires.py


--- :bookmark: example data for demonstration ---

example input data in: example_extracted_data/

:warning: IMPORTANT: LISCENCE REQUIRED FOR FORMAL USE OF THE DATA! 

--- :sparkles:	major tunable hyperparameters ---

- boundary weights: determine the penalty on crown boundaries for better individual separation

- task_ratio: the ratio for weighting the two losses to balance the emphasis on both tasks during training, may differ from case to case

- normalize: ratio to augment data by randomly normalizing image patches 

-------------------------------------------------------------------------------------------

### Train 2nd model: Tree height estimation from aerial images:

```
python main2_height_prediction.py
```

--- :bookmark: set configs ---

config/UNetTraining_CHM.py


-------------------------------------------------------------------------------------------

### Test 1st model: segmentation & counting:

```
python step3_predict_segmentation_counting.py
```

--- :bookmark: set configs ---

config/Model_compare_multires.py

--- :flags: Example prediction ---

See /example_extracted_data/

- segmentation result: seg_41.tif

- counting result: density_41.tif

:warning: Note that the model was trained using image patch no.41, and thus should not be tested using the same image in the test phrase. Here we simply demonstrate how to apply the model on a test image.

------------------------------------------------------------------------------------------------


##
![Figure 1](figures/fig1.png)

Figure 1. Overview of the framework used to count individual trees and predict their crown area and height. a, Deep learning-based framework for individual tree counting, crown segmentation, and height prediction. Spatial locations of individual trees are incorporated in the tree density maps and the crown segmentation maps. The canopy height map (CHM) derived from LiDAR data provides pixel-wise height information, which, when available for a specific study area, can optionally be used as an additional input band for the individual tree counting and crown segmentation tasks. b, Data preparation and modeling for tree counting and crown segmentation. The manually delineated individual tree crowns are modeled as density maps for the counting task by extracting the polygon centroids. The gaps between adjacent crowns are highlighted for the separation of individual tree crowns during the training phase. <br />

##
![Figure 2](figures/fig2.png)

Figure 2. Example products from the proposed framework. a, Wall-to-wall tree count prediction for Denmark. b, Detailed examples showing the individual tree counting (second row), crown segmentation (third row), and height prediction (third row) from three major types of landscapes (deciduous forest, coniferous forest, and non-forest). c, Large-scale individual tree crown segmentation results colored by height predictions. Examples in b and c were sampled from the region indicated by the orange box in a.




