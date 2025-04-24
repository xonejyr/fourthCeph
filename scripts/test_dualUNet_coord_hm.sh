
# for heatmap
./scripts/validate.sh ./configs/256x256_dualunet_ce_heatmap.yaml ./exp/train_ce_hm_DualUNet-256x256_dualunet_ce_heatmap/best.pth train_ce_hm_DualUNet &

# for coord
./scripts/validate.sh ./configs/256x256_dualunet_ce_coord.yaml ./exp/train_ce_coord_DualUNet-256x256_dualunet_ce_coord/best.pth train_ce_coord_DualUNet

echo "- Exp-test for DualUNet is done for both coord and heatmap"