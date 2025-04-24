### Train on X-ray Cephalograms with Hm. Unet
``` bash
./scripts/train.sh ./configs/512x512_unet_ce_heatmap.yaml train_ce_hm_unet

./scripts/train.sh ./configs/256x256_dualunet_ce_heatmap.yaml train_ce_hm_DualUNet

./scripts/train_1_1.sh ./configs/512x512_ResUNetFPN_ce_heatmap.yaml train_ce_hm_ResUNetFPN

./scripts/train_1_1.sh ./configs/256x256_unetFPN_ce_heatmap.yaml train_ce_hm_UNetFPN

###############################################
./scripts/train_1_1.sh ./configs/512x512_NFDP_ce_heatmap_RLE.yaml train_ce_hm_NFDP
./scripts/train_1_1.sh ./configs/512x512_NFDP_ce_heatmap_RLE_noRes.yaml train_ce_hm_NFDP
./scripts/train_and_val.sh ./configs/512x512_NFDP_ce_heatmap_noPretrained.yaml train_ce_hm_NFDP

./scripts/train_and_val.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE_noRes.yaml train_ce_hm_HeatmapBasisNFR
./scripts/train_and_val.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFR

./scripts/train_and_val.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE_numJoints.yaml train_ce_hm_HeatmapBasisNFR

./scripts/train_and_val.sh ./configs/512x512_HeatmapBasisNFRDynamic_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFRDynamic

bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFR 3 > search-HeatmapBasisNFR-20250423.log 2>&1

bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE_numJoints.yaml train_ce_hm_HeatmapBasisNFR 3 > search-heatmap_RLE_numJoints-20250423.log 2>&1

nohup bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFRDynamic_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFRDynamic 3 > search-HeatmapBasisNFRDynamic-20250423.log 2>&1

./scripts/train_and_val.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE_FC=512256.yaml train_ce_hm_HeatmapBasisNFR
nohup your_command > /dev/null 2>&1 & #不保留结果
###############################################
./scripts/run_analysis.sh 


./scripts/train_and_val.sh ./configs/512x512_NFDP_ce_heatmap_test.yaml tv_test
./scripts/utils/param_search/param_search.sh ./configs/512x512_NFDP_ce_heatmap_test.yaml tv_test 2 ./configs/search_yamls/search-tv_test-512x512_NFDP_ce_heatmap_test-20250423152459.yaml
###############################################


./scripts/train_1_1.sh ./configs/512x512_NFDP_ce_heatmap.yaml train_ce_hm_NFDP

./scripts/train_1_1.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap.yaml train_ce_hm_HeatmapBasisNFR

./scripts/train_1_1.sh ./configs/512x512_NFDP_ce_heatmap-noNF.yaml train_ce_hm_NFDP

./scripts/train_1_1.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap-noBase.yaml train_ce_hm_HeatmapBasisNFR

./scripts/train_1_1.sh ./configs/512x512_HeatmapBasisNFRwithOrth_ce_heatmap.yaml train_ce_hm_HeatmapBasisNFRwithOrth

./scripts/train_1_1.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_mseloss.yaml train_ce_hm_HeatmapBasisNFR

./scripts/train_1_1.sh ./configs/512x512_Hybrid_ce_heatmap.yaml train_ce_hm_Hybird

./scripts/train_1_1.sh ./configs/512x512_HybridUResFPN_ce_heatmap.yaml train_ce_hm_HybridUResFPN

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPN_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPN

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNEnhanced_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNEnhanced

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNEnhancedMultiStep_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNEnhancedMultiStep

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNEnhancedMultiStep_ceRebuild_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNEnhancedMultiStep


./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNEnhancedMultiStepSample_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNEnhancedMultiStepSample

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNMultichannel_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNMultichannel

./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPNOnlyPointDistance_embedding_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNOnlyPointDistance

./scripts/train_1_1.sh ./configs/512x512_ResFPN_UNet512_ce_heatmap.yaml train_ce_hm_ResFPN_UNet512

./scripts/train_1_1.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap.yaml train_ce_hm_ResFPN_HeatmapBasisNFR


./scripts/train_1_1.sh ./configs/256x256_ResFPN_ce_heatmap.yaml train_ce_hm_ResFPN

./scripts/train_1_2.sh ./configs/256x256_ResFPN_ce_heatmap.yaml train_ce_hm_ResFPN #bad for sub-pixel
```

### Validate on X-ray Cephalograms with Hm. Unet
``` bash
./scripts/validate.sh ./configs/512x512_unet_ce_heatmap.yaml ./exp/train_ce_hm_unet-512x512_unet_ce_heatmap/best.pth train_ce_hm_unet

./scripts/validate.sh ./configs/256x256_ResFPN_ce_heatmap.yaml ./exp/train_ce_hm_ResFPN-256x256_ResFPN_ce_heatmap/best.pth train_ce_hm_ResFPN

 ./scripts/validate.sh ./configs/512x512_NFDP_ce_heatmap.yaml ./exp/train_ce_hm_NFDP-512x512_NFDP_ce_heatmap/best.pth train_ce_hm_NFDP

./scripts/validate.sh ./configs/512x512_Hybrid_ce_heatmap.yaml ./exp/train_ce_hm_Hybird-512x512_Hybrid_ce_heatmap/best.pth train_ce_hm_Hybird

./scripts/validate.sh  ./configs/512x512_HierarchicalGraphResFPN_ce_heatmap.yaml ./exp/train_ce_hm_HierarchicalGraphResFPN-512x512_HierarchicalGraphResFPN_ce_heatmap/best.pth  train_ce_hm_HierarchicalGraphResFPN


./scripts/validate.sh  ./configs/512x512_HierarchicalGraphResFPNEnhanced_ce_heatmap.yaml /mnt/home_extend/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_HierarchicalGraphResFPNEnhanced-512x512_HierarchicalGraphResFPNEnhanced_ce_heatmap/params_search/models/model_dbb1fcf8-bb20-4b84-bdd9-1a4e3ee02edf/best.pth  train_ce_hm_HierarchicalGraphResFPNEnhanced

./scripts/validate.sh ./configs/256x256_ResFPN_ce_heatmap.yaml /mnt/home_extend/python/vscode/Jingyu/Landmark/secondCeph/exp/train_ce_hm_ResFPN-256x256_ResFPN_ce_heatmap/params_search-ResNet_SIZE_LR_TYPE/models/model_ac2e9084-1cc2-4d22-85b8-cf7e1ae44a0e/best.pth train_ce_hm_ResFPN

# using the specific pth to validate the exp "train_ce_hm_unet"
./scripts/validate.sh ./configs/512x512_unet_ce_heatmap.yaml ./exp/train_ce_hm_unet-512x512_unet_ce_heatmap_20250312/best.pth train_ce_hm_unet

./scripts/validate.sh ./configs/512x512_unet_ce_heatmap.yaml ./exp/train_ce_hm_unet-512x512_unet_ce_heatmap/best_model_9294b_00090.pth train_ce_hm_unet
```

### Train on X-ray Cephalograms with coord. Unet
``` bash
./scripts/train.sh ./configs/512x512_unet_ce_coord.yaml train_ce_coord_unet

./scripts/train.sh ./configs/256x256_DHDN_ce_coord.yaml train_ce_coord_DHDN 

./scripts/train.sh ./configs/256x256_DHDN_ce_coord_300.yaml train_ce_coord_DHDN 

./scripts/train_1_1.sh ./configs/256x256_dualunet_ce_coord.yaml train_ce_coord_DualUNet
```
### Validate on X-ray Cephalograms with coord. Unet
``` bash
./scripts/validate.sh ./configs/512x512_unet_ce_coord.yaml ./exp/train_ce_coord_unet-512x512_unet_ce_coord/final.pth train_ce_coord_unet

# using the specific pth to validate the exp "train_ce_coord_unet"
./scripts/validate.sh ./configs/512x512_unet_ce_coord.yaml ./exp/train_ce_coord_unet-512x512_unet_ce_coord_20250228/best.pth train_ce_coord_unet
```

### hyperparams searching
```bash
# param search for train_ce_hm_unet or train_ce_coord_unet with specific number of exps
# the first number for how many exps do at the mean time, defualt as 1
# the second number refers to the total number of the exps
bash ./scripts/utils/param_search/param_search.sh configs/512x512_unet_ce_heatmap.yaml train_ce_hm_unet 4 8 
bash ./scripts/utils/param_search/param_search.sh configs/512x512_unet_ce_coord.yaml train_ce_coord_unet 4 18
bash ./scripts/utils/param_search/param_search.sh configs/256x256_DHDN_ce_coord.yaml train_ce_coord_DHDN 2 768

bash ./scripts/utils/param_search/param_search.sh configs/256x256_dualunet_ce_heatmap.yaml train_ce_hm_DualUNet 2 4
bash ./scripts/utils/param_search/param_search.sh configs/256x256_dualunet_ce_heatmap.yaml train_ce_hm_DualUNet 2 1
bash ./scripts/utils/param_search/param_search.sh configs/256x256_dualunet_ce_heatmap.yaml train_ce_hm_DualUNet 3

bash ./scripts/utils/param_search/param_search.sh configs/256x256_ResFPN_ce_heatmap.yaml train_ce_hm_ResFPN 2

bash ./scripts/utils/param_search/param_search.sh configs/512x512_HeatmapBasisNFR_ce_heatmap.yaml train_ce_hm_HeatmapBasisNFR 4 > HeatmapBasisNFR.log 2>&1 &

bash ./scripts/utils/param_search/param_search.sh configs/512x512_HierarchicalGraphResFPNEnhanced_ce_heatmap.yaml train_ce_hm_HierarchicalGraphResFPNEnhanced 1




# analysis the importance of the params to performance (need to settings at first)
python ./scripts/utils/importance_analysis.py

# get the top items of the ray.results, and Box plot by-parameter and global scatter plot (need to settings at first)
python ./scripts/utils/load_analysis.py

# get the landmarks distribution of IMAGE_SIZE for heatmap-based method (need to settings at first)
python ./scripts/utils/plot_landmarks_distribution.py --cfg ./configs/512x512_unet_ce_heatmap.yaml

# plot the train history from csv files of loss and metrics
python ./scripts/utils/plot_train_history.py

# get the metrics by-landmark, and the mre excel results for image-landmark (need to settings at first) 
python ./scripts/utils/get_pts_statistics.py

# put the gt_pts and pred_tps on the raw Image (need to settings at first)
python ./scripts/utils/draw_pts_img.py

# visualize the model
python ./scripts/utils/model_visualize.py

# test the shared_params_manage.py
python ./scripts/utils/shared_params_manage.py
```

# My test for DualUNet
```bash
# for heatmap
./scripts/train_1_1.sh ./configs/256x256_dualunet_ce_heatmap.yaml train_ce_hm_DualUNet

# for coord
./scripts/train_1_1.sh ./configs/256x256_dualunet_ce_coord.yaml train_ce_coord_DualUNet

# for both
./scripts/train_dualUNet_coord_hm.sh
./scripts/test_dualUNet_coord_hm.sh
```