wget https://packages.gurobi.com/9.1/gurobi9.1.1_linux64.tar.gz
mv gurobi9.1.1_linux64.tar.gz /opt
cd /opt
tar xvfz /opt/gurobi9.1.1_linux64.tar.gz

# Add export variables to ~/.bashrc 
echo 'export GUROBI_HOME="/opt/gurobi911/linux64"' >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc