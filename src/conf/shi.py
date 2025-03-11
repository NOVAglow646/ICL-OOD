import yaml
with open('/data1/qxwang/codes/in-context-learning/src/conf/autoregression.yaml', "w") as yaml_file:
    yaml.dump(args.__dict__, yaml_file, default_flow_style=False)