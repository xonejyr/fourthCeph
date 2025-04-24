EXP_NAME=$(python3 -c "import sys; sys.path.append('./scripts/utils'); from shared_params_manage import ParamManager; pm = ParamManager(); print(pm.get_experiment_name())")

echo $EXP_NAM
# Get the Latex Table
echo "Get the Latex Table..."
./scripts/utils/generate_results_table.sh "exp/$EXP_NAME" -1