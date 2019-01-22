URL=https://www.dropbox.com/s/zpo35fq1vafe637/gpgan_lfw.zip?dl=0
ZIP_FILE=./gpgan_lfw.zip
TARGET_DIR=./
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $TARGET_DIR
rm $ZIP_FILE
