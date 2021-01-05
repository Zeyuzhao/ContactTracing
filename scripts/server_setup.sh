# Before running, be sure to have git and conda installed
# Execute this script in an empty folder

# Install and setup lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone repository
git clone https://github.com/Zeyuzhao/ContactTracing.git

# Enter correct branch
cd ContactTracing
git checkout main

# Setup environment
conda env create -f environment.yml
conda activate tracing

# Execute scripts
cd scripts
python parallel.py
