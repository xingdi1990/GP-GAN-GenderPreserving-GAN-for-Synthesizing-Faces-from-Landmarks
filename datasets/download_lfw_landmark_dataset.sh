URL=https://www.dropbox.com/s/45tu9jabvo0hkq1/lfw_gpgan.tar.gz?dl=0
ZIP_FILE=./lfw_gpgan.tar.gz
TARGET_DIR=./lfw/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
tar -xzvf $ZIP_FILE
rm $ZIP_FILE
