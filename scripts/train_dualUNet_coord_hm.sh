
# for heatmap
./scripts/train_1_1.sh ./configs/512x512_HierarchicalGraphResFPN_ce_3_heatmap.yaml train_ce_hm_HierarchicalGraphResFPN &

# for coord
./scripts/train_1_1.sh ./configs/512x512_ResFPN_ce_3_heatmap.yaml train_ce_hm_ResFPN

echo "the exp for DualUNet is done for both coord and heatmap"