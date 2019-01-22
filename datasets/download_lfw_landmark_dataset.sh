URL=https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/xdi1_jh_edu/EQgp0DIRW1FAieMlKFL6W5wBg95Z8JVcq2YDmQpNZagSRg?e=xklBw7
ZIP_FILE=./gpgan_lfw.tar.xz
TARGET_DIR=./lfw/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
tar -xzvf $ZIP_FILE
rm $ZIP_FILE
