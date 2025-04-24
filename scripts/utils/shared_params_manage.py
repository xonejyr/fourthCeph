import os
import yaml
from ray import tune
from itertools import product

class ParamManager:
    def __init__(self, config_file="./scripts/utils/shared_params.yaml"):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base = self.config["base"]
        self.root_dir = self.base["root_dir"]
        self.experiment_name = self.base["experiment_name"]
        self.val_dataset = self.base["val_dataset"]
        self.model_visualization_dir = self.base["model_visualization_dir"]
        self.sort_by_importance = self.config.get('sort_by_importance', [])


        if self.val_dataset == "val":
            self.data_dir = "data/ISBI2015/RawImage/Test1Data"
            self.json_file = "val_gt_kpt.json"
        elif self.val_dataset == "test":
            self.data_dir = "data/ISBI2015/RawImage/Test2Data"
            self.json_file = "test_gt_kpt.json"
        else:
            raise ValueError("Invalid val_dataset, should be 'val' or 'test'")
        
        # 自动生成路径
        self.json_path = os.path.join(self.root_dir, "exp", self.experiment_name, self.json_file)
        self.image_dir = os.path.join(self.root_dir, self.data_dir)
        self.pts_output_dir = os.path.join(self.root_dir, "exp", self.experiment_name, "pts_img")
        self.experiment_path = os.path.join(self.root_dir, "exp", self.experiment_name, "params_search/metadata_for_search")
        self.experiment_fig_dir = os.path.join(self.root_dir, "exp", self.experiment_name, "params_search/visualizations")
        self.root_fig_dir = os.path.join(self.root_dir, "exp", self.experiment_name, "visualizations")
        self.model_visualization_dir = os.path.join(self.model_visualization_dir)
        self.csv_path = os.path.join(self.root_dir, "exp", self.experiment_name, "train_history.csv")
        
        # 自动从 search_space 生成并存储 cfg_path
        self.search_space, self.cfg_paths = self._parse_search_space()
        self.parameter_columns = list(self.config["search_space"].keys())
    
        self.dependent_params = self._get_dependent_parameters()


    def _parse_search_space(self):
        raw_space = self.config["search_space"]
        constraints = self.config.get("constraints", [])
        cfg_paths = {}
    
        # 提取 cfg_path
        for param, spec in raw_space.items():
            if spec["type"] == "choice":
                cfg_paths[param] = spec["cfg_path"]
    
        # 分离独立参数和依赖参数
        independent_params = {}
        dependent_params = {}
        depends_on_mapping = {}
    
        # 收集依赖参数和约束映射
        if constraints:
            for constraint in constraints:
                dep_param = constraint["param"]
                depends_on = constraint["depends_on"]
                dependent_params[dep_param] = raw_space[dep_param]["values"]
                # 将字符串键转换为 tuple
                mapping = {}
                for k, v in constraint["mapping"].items():
                    # 假设键是字符串如 "[512,512]"，转换为 tuple
                    key_tuple = tuple(map(int, k.strip("[]").split(",")))
                    mapping[key_tuple] = v
                depends_on_mapping[dep_param] = {
                    "depends_on": depends_on,
                    "mapping": mapping
                }
    
        # 收集独立参数
        for param, spec in raw_space.items():
            if param not in dependent_params:
                independent_params[param] = spec["values"]
    
        # 生成独立参数的所有组合
        independent_keys = list(independent_params.keys())
        independent_values = [independent_params[k] for k in independent_keys]
        base_configs = [dict(zip(independent_keys, combo)) for combo in product(*independent_values)]
    
        # 生成所有有效组合
        all_combinations = []
        for base_config in base_configs:
            configs_to_process = [base_config.copy()]
            for dep_param, mapping_info in depends_on_mapping.items():
                depends_on = mapping_info["depends_on"]
                mapping = mapping_info["mapping"]
                new_configs = []
    
                for config in configs_to_process:
                    dep_value = tuple(config[depends_on]) if isinstance(config[depends_on], (list, tuple)) else config[depends_on]
                    allowed_values = mapping.get(dep_value, None)
    
                    if allowed_values is None:
                        continue
                    for value in allowed_values:
                        new_config = config.copy()
                        new_config[dep_param] = value if isinstance(value, (list, tuple)) else value
                        new_configs.append(new_config)
                configs_to_process = new_configs
    
            all_combinations.extend(configs_to_process)
    
        search_space = tune.grid_search(all_combinations)
        return search_space, cfg_paths
        
    def get_paths(self):
        """返回所有生成的路径"""
        return {
            "json_path": self.json_path,
            "image_dir": self.image_dir,
            "pts_output_dir": self.pts_output_dir,
            "experiment_path": self.experiment_path,
            "experiment_fig_dir": self.experiment_fig_dir,
            "root_fig_dir": self.root_fig_dir,
            "model_visualization_dir": self.model_visualization_dir,
            "csv_path": self.csv_path
        }
    
    def get_search_space(self):
        return self.search_space
    
    def get_experiment_path(self):
        return self.experiment_path
    
    def get_experiment_name(self):
        return self.experiment_name
    
    def get_parameter_columns(self):
        return self.parameter_columns
    

    
    def infer_types(self):
        cat_columns = []
        num_columns = []
        for param in self.parameter_columns:
            values = self.config["search_space"][param]["values"]
            # 检查所有值的类型
            all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None)
            all_strings_or_lists = all(isinstance(v, (str, list)) for v in values if v is not None)
            
            if all_numeric:
                num_columns.append(f"{param}")
            elif all_strings_or_lists or any(v is None for v in values) or any(isinstance(v, bool) for v in values):
                cat_columns.append(f"{param}")
            else:
                # 如果类型混合严重，打印警告
                print(f"警告：参数 {param} 的值 {values} 类型不一致，归为分类变量")
                cat_columns.append(f"{param}")
        
        return cat_columns, num_columns
        
    def update_train_config(self, config, cfg):
        """通过 cfg_path 自动更新 cfg 中的值"""
        for param, value in config.items():
            if param in self.cfg_paths:
                path = self.cfg_paths[param].split('.')
                current = cfg
                # 遍历路径，直到倒数第二级
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                # 设置最后一级的值
                current[path[-1]] = value
        return cfg
    
    def _get_dependent_parameters(self):
        constraints = self.config.get("constraints", [])
        return [constraint["param"] for constraint in constraints]
    
    def get_cfg_paths(self):
        return self.cfg_paths
    


if __name__ == "__main__":

    param_manager = ParamManager()
    paths = param_manager.get_paths()
    search_space = param_manager.get_search_space()
    parameter_columns = param_manager.get_parameter_columns()
    infer_types = param_manager.infer_types()
    print("Paths:", paths)
    print("Search Space:", search_space)
    
    print(f"The size of search space is: {len(search_space['grid_search'])}")
    print("parameter Columns:", parameter_columns)
    print("Infer Types:", infer_types)

    cfg_paths = param_manager.get_cfg_paths()
    print("Cfg Paths:", cfg_paths)

    independent_params = param_manager.dependent_params
    print("Dependent Parameters:", independent_params)

    #print(f"All samplers are: {param_manager.all_combinations}")
    #print(f"The number of all combinations is: {len(param_manager.all_combinations)}")

    ## 测试 update_train_config
    #cfg = {"DATASET": {"PRESET": {"METHOD_TYPE":{}}}, "MODEL": {}, "LOSS": {}}
    #print("\nOriginal CFG:", cfg)
    #sample_config = {"METHOD_TYPE": "heatmap", "LOSS_TYPE": "AGD2UNetLoss"}
    #updated_cfg = param_manager.update_train_config(sample_config, cfg)
    #print("\nUpdated CFG:", updated_cfg)
#
    #search_params={}
    #for param, cfg_path in param_manager.cfg_paths.items():
    #    path = cfg_path.split('.')
    #    current = cfg
    #    for key in path:
    #        current = current[key]
    #    search_params[param] = current
    #print("\nSearch Parameters:", search_params)
