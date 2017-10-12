# -*- coding:utf-8 -*-
import train
import download
import input


def main():
    download.download_dir(download.HDFS_TRAIN_PATH,download.LOCAL_TRAIN_PATH)
    download.download_dir(download.HDFS_TEST_PATH,download.LOCAL_TEST_PATH)
    #
    input.gen_train_tf()
    input.gen_test_tf()
    #
    train.main()


if __name__ == '__main__':
    main()