#!/bin/bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> /var/folders/3_/dwwjz7qn6sl1_dzxbr_151wr0000gn/T/vscode-zsh/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /var/folders/3_/dwwjz7qn6sl1_dzxbr_151wr0000gn/T/vscode-zsh/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install python@3.10
python3.10 -m venv octis_env
source octis_env/bin/activate
pip3 install numpy
pip3 install sentence_transformers
pip3 install umap
pip3 install hdbscan
pip3 install matplotlib
pip3 install seaborn
pip3 install octis
pip3 install umap-learn
