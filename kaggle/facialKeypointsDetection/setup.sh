sudo pip uninstall matplotlib
sudo apt-get install python-matplotlib
sudo pip install pandas numpy keras

pip install --upgrade git+git://github.com/Theano/Theano.git

wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip
wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip
mv www.kaggle.com/c/facial-keypoints-detection/download/* .
sudo apt-get install unzip
unzip training.zip
unzip test.zip
rm -rf www.kaggle.com/