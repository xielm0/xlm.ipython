# -*- coding:utf-8 -*-

import download
import input
import apply


def main():
    download.download_dir(download.HDFS_APPLY_PATH,download.LOCAL_APPLY_PATH,2)
    #
    apply.main()


if __name__ == '__main__':
    main()