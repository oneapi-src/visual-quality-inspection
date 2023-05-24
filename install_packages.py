import yaml
import os

def install_packages_from_yml(file_path):
    with open(file_path, 'r') as stream:
        try:
            env_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Regular dependencies
    for dep in env_data['dependencies']:
        if isinstance(dep, str):
            if "python=" in dep:  # Python version specification, ignore
                continue
            if "pip=" in dep:  # Pip version specification, ignore
                continue
            if "==" in dep:  # Version specified
                package = dep.split('=')[0]
                version = dep.split('=')[1]
                os.system(f"pip install {package}=={version}")
            else:
                package = dep
                os.system(f"pip install {package}")
    
    # Pip dependencies
    if 'pip' in env_data['dependencies'][-1]:
        pip_packages = env_data['dependencies'][-1]['pip']
        for pip_package in pip_packages:
            if "==" in pip_package:  # Version specified
                package = pip_package.split('==')[0]
                version = pip_package.split('==')[1]
                os.system(f"pip install {package}=={version}")
            else:  # No version specified
                os.system(f"pip install {pip_package}")


# Use function to install packages from yml files
install_packages_from_yml('/dbfs/FileStore/tables/visual-quality-inspection-main/env/stock/stock_pytorch.yml')
install_packages_from_yml('/dbfs/FileStore/tables/visual-quality-inspection-main/env/intel/aikit_pt.yml')