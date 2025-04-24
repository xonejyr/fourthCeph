echo "[1/3] Running HeatmapBasisNFR"
bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFR 3 > search-HeatmapBasisNFR-20250423.log 2>&1 && \

echo "[3/3] Running HeatmapBasisNFRDynamic (nohup & background)" bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFRDynamic_ce_heatmap_RLE.yaml train_ce_hm_HeatmapBasisNFRDynamic 3 > search-HeatmapBasisNFRDynamic-20250423.log 2>&1 && \

echo "[2/3] Running HeatmapBasisNFR_numJoints"
bash ./scripts/utils/param_search/param_search.sh ./configs/512x512_HeatmapBasisNFR_ce_heatmap_RLE_numJoints.yaml train_ce_hm_HeatmapBasisNFR 3 > search-heatmap_RLE_numJoints-20250423.log 2>&1 && 


