wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
sudo mv bfg-1.14.0.jar /opt/bfg.jar

echo 'alias bfg="java -jar /opt/bfg.jar"' >> ~/.bashrc
source ~/.bashrc

