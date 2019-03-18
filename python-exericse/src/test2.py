import os, sys

sys.path.insert(0, '../bin')
sys.path.insert(0, '../conf')
sys.path.insert(0, '../model_team')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/custom_cmds')


def sync_model_exp():
    # sync model for experiment version, not supported for online.
    import logging, time, glob
    from env import serving_ip, serving_username, serving_password, serving_model_dst
    ip = serving_ip
    username = serving_username
    password = serving_password
    model_dst = serving_model_dst
    data_path = '../data'
    logging.info('%s %s %s %s', ip, username, password, model_dst)

    path = '../models/'
    if path.endswith('/'):
        path = path[:-1]
    all_model_path = glob.glob(path +
                               '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/')
    not_null_model_path = [
        x for x in all_model_path if os.path.exists(x + '/export')
    ]
    if len(not_null_model_path) != 1:
        logging.info('[F] not exactly one model in ../models folder.')
        raise Exception('SYNC Error')
    model_path = not_null_model_path[0]

    logging.info('Found model: %s', model_path)
    logging.info('model path: %s', path)
    # delete old models
    cache_model_num = 7
    models_to_delete = sorted(glob.glob(path + '/_1*'), reverse=True)[cache_model_num - 1:]
    if len(models_to_delete) > 0:
        delete_model_cmd = '/bin/rm -rf %s' % (' '.join(models_to_delete))
        logging.info(delete_model_cmd)
        os.system(delete_model_cmd)

    rename_name = 'tf_models'
    backup_name = path + '/_' + str(int(time.time()))

    tag_model_cmd = 'mv %s %s' % (model_path, backup_name)
    rename_cmd = 'cd %s && rm %s* ; ln -s %s %s && cd -' % (path, rename_name, backup_name, rename_name)
    md5sum_cmd = 'cd %s && md5sum %s/* > %s.status && cd -' % (path, rename_name, rename_name)
    sync_model_cmd = './.sshpass -p "%s" scp -r %s/%s* %s@%s:%s' % (
    password, path, rename_name, username, ip, model_dst)
    sync_data_cmd = './.sshpass -p "%s" scp -r %s %s@%s:%s' % (password, data_path, username, ip, model_dst)

    logging.info(tag_model_cmd)
    os.system(tag_model_cmd)
    logging.info(rename_cmd)
    os.system(rename_cmd)
    logging.info(md5sum_cmd)
    os.system(md5sum_cmd)
    logging.info(sync_model_cmd)
    os.system(sync_model_cmd)
    logging.info(sync_data_cmd)
    os.system(sync_data_cmd)


def prepare_norm_guard_data():
    # hdfs_path = train_configure['hdfs_path']
    # cmd = "hadoop fs -cat `hadoop fs -ls -h /user/jd_ad/ads_reco/detail_re_ads_expand_data/gen_pair |awk '{print $8}'|tail -n 1`/*/* | head -n 50000 > .local_data"
    cmd = "hadoop fs -cat `hadoop fs -ls -h /user/jd_ad/ads_conv/unify_model_train_data/midpage_pairwise_data/ |awk '{print $8}'|tail -n 1`/*/* | head -n 50000 > .local_data"
    os.system(cmd)
    import time, datetime
    current_date = time.strftime('%Y%m%d%H%M', time.localtime())
    os.system("sed -i 's/\\t//g' .local_data")
    os.system("sed -i 's/1 1|/1\\t1|/g' .local_data && sed -i 's/0 1|/0\\t1|/g' .local_data")
    os.system("sed -i 's/|||/\\n/g' .local_data && mv .local_data %s" % current_date)
    return current_date


def send_model():
    '''send model and test data to norm-gard.'''
    import logging, glob
    # rename_name should match the conf in predictor.
    rename_name = 'tf_models'
    data_path = '../data'
    # model_dst is the path in norm_gard to send to.
    model_dst = ''
    with open('../model.des', 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] == '#':
                continue
            key, value = line.split('=', 1)
            if key == 'destination':
                model_dst = value
                break
        else:
            logging.info('[F] Can\'t read destination folder,please check model.des.')
            raise Exception('Conf Error')
    # get dumped model folder.
    path = '../models/'
    if path.endswith('/'):
        path = path[:-1]
    all_model_path = glob.glob(path +
                               '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/')
    not_null_model_path = [
        x for x in all_model_path if os.path.exists(x + '/export')
    ]
    if len(not_null_model_path) != 1:
        logging.info('[F] not exactly one model in ../models folder.')
        raise Exception('SYNC Error')
    model_path = not_null_model_path[0]
    logging.info('Found model: %s', model_path)
    logging.info('model path: %s', path)
    # do them
    rename_cmd = 'cd %s && rm %s ; ln -s %s %s && cd -' % (path, rename_name, model_path, rename_name)
    renamed_path = os.path.join(os.path.dirname(model_path[:-1]), rename_name)
    md5sum_cmd = 'cd %s && md5sum %s/* > %s.status && cd -' % (path, rename_name, rename_name)

    import time, datetime
    datetime_path = time.strftime('%Y%m%d%H%M', time.localtime())
    mfs_data_path = '%s/data/%s' % (model_dst, datetime_path)
    mfs_model_path = '%s/model/%s' % (model_dst, datetime_path)
    sync_model_cmd = 'mkdir -p %s/model && cp -rLf %s* %s/model' % (mfs_model_path, renamed_path, mfs_model_path)
    sync_data_cmd = 'cp -rf %s %s' % (data_path, mfs_model_path)
    # norm_guard_data_path = prepare_norm_guard_data()
    # sync_norm_guard_data_cmd = 'mv -f %s %s' % (norm_guard_data_path, mfs_data_path)
    backup_model_cmd = 'cd %s && mv %s ../models_backup/ && cd -' % (path, model_path)
    touch_success_flag = 'touch %s/_TRANSFER_DONE' % mfs_model_path
    chmod_cmd = 'chmod -R 777 %s %s' % (mfs_model_path, mfs_model_path)

    logging.info(rename_cmd)
    os.system(rename_cmd)
    logging.info(md5sum_cmd)
    os.system(md5sum_cmd)
    logging.info(sync_model_cmd)
    os.system(sync_model_cmd)

    logging.info(sync_data_cmd)
    os.system(sync_data_cmd)

    # logging.info(sync_norm_guard_data_cmd)
    # os.system(sync_norm_guard_data_cmd)

    logging.info(backup_model_cmd)
    os.system(backup_model_cmd)

    logging.info(touch_success_flag)
    os.system(touch_success_flag)

    logging.info(chmod_cmd)
    os.system(chmod_cmd)


send_model()
# ==============
from custom_cmds.cmds import BackupLogs

BackupLogs()


def mvlogs():
    import logging, time
    datetime_path = time.strftime('%Y%m%d%H%M', time.localtime())
    folder = '/tmp/app_home_unify/' + datetime_path
    cmd = 'cp -r ../logs ' + folder

    logging.info(cmd)
    os.system(cmd)

# mvlogs()

