### Train on X-ray Cephalograms with Hm. Unet
``` bash
nohup your_command > /dev/null 2>&1 & #不保留结果
###############################################
./scripts/train_and_val.sh ./configs/512x512_NFDP_ce_heatmap.yaml tv_test

./scripts/utils/param_search/param_search.sh ./configs/512x512_NFDP_ce_heatmap_test.yaml tv_test 2 ./configs/search_yamls/search-tv_test-512x512_NFDP_ce_heatmap-val.yaml
###############################################

###############################################
# test
./scripts/train_and_val.sh ./configs/512x512_res18_ce_heatmap_NFDP.yaml train_NFDP
./scripts/train_and_val.sh ./configs/512x512_res18_ce_heatmap_NFDP_logQ.yaml train_NFDP
./scripts/train_and_val.sh ./configs/512x512_res18_ce_heatmap_NFDP_noRes.yaml train_NFDP
./scripts/train_and_val.sh ./configs/512x512_res18_ce_heatmap_NFDP_lrArticle.yaml train_NFDP
./scripts/train_and_val.sh ./configs/512x512_res18_ce_heatmap_NFDP_lr001.yaml train_NFDP



./scripts/utils/param_search/param_search.sh ./configs/512x512_NFDP_ce_heatmap.yaml tv_test 2 ./configs/search_yamls/train_NFDP-512x512_NFDP_ce_heatmap-val.yaml
###############################################



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


####################################################################
# 2025.5.26 test for L1 and L2
```bash
./scripts/visualization_pred_heatmap.sh ./configs/512x512_res18_ce_heatmap_NFDP_again.yaml ./exp/train_NFDP-512x512_res18_ce_heatmap_NFDP_again/best.pth train_NFDP


nohup ./scripts/train_and_val.sh ./configs/512x512_res18_ce_coord_NFDP_mynoise.yaml train_NFDP > logs/512x512_res18_ce_coord_NFDP_mynoise.log 2>&1 &

nohup ./scripts/train_and_val.sh ./configs/512x512_res18_ce_coord_NFDP_LFPnoise.yaml train_NFDP > logs/512x512_res18_ce_coord_NFDP_LFPnoise.log 2>&1 &
``` 