# -*- coding:utf-8 -*-
import os, sys

sys.path.insert(0, '../bin')
sys.path.insert(0, '../conf')
sys.path.insert(0, '../model_team')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/custom_cmds')

from custom_cmds.cmds import BackupEnv
from custom_cmds.cmds import BackupLogs
import paramiko

print('This is sample script run after train.')



def prepare_norm_guard_data(filename):
    # hdfs_path = train_configure['hdfs_path']
    cmd = "hadoop fs -cat `hadoop fs -ls -h /user/jd_ad/ads_sz/app.db/app_szad_m_midpage_pointwise_data_train |awk '{print $8}'|tail -n 1`/*/* | head -n 50000 > .local_data"
    os.system(cmd)
    import time, datetime
    os.system("sed -i 's/|||/\\n/g' .local_data && mv .local_data %s" % filename)


def send_dir(sftp, sour, dst):
    dst_dir = os.path.join(dst, sour.split('/')[-1])
    sftp.mkdir(dst_dir)
    for ddir in os.walk(sour):
        root = ddir[0]
        f_dir = ddir[1]
        file = ddir[2]
        root2 = root.split('/', 2)[-1]
        for d in f_dir:
            sftp.mkdir(os.path.join(dst, root2, d))
        for f in file:
            sftp.put(os.path.join(root, f), os.path.join(dst, root2, f))


def send_norm_guard_data(sftp, sour, dst):
    sftp.put(sour, dst)


def scp_model_dir(local_path, norm_guard_data_path, model_dst, logging):
    logging.info('using paramiko to sync the model with the MFS')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('172.19.178.31', username='admin', password='0okm(IJN', timeout=9999)
    sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    send_dir(sftp, local_path, os.path.join(model_dst, 'model'))
    send_norm_guard_data(sftp, norm_guard_data_path, os.path.join(model_dst, 'data', norm_guard_data_path))
    # send a _TRANSFER_DONE tag
    transfer_done = '_TRANSFER_DONE'
    finish_cmd = 'touch %s' % transfer_done
    logging.info(finish_cmd)
    os.system(finish_cmd)
    sftp.put(transfer_done, os.path.join(model_dst, 'model', norm_guard_data_path, transfer_done))
    logging.info('finished transforming model')
    ssh.exec_command('chmod 777 -R %s' % os.path.join(model_dst, 'model', norm_guard_data_path))
    logging.info('chmod 777 -R %s' % os.path.join(model_dst, 'model', norm_guard_data_path))
    ssh.exec_command('chmod 777 %s' % os.path.join(model_dst, 'data', norm_guard_data_path))
    logging.info('chmod 777 %s' % os.path.join(model_dst, 'data', norm_guard_data_path))


def send_model():
    '''send model and test data to norm-gard.'''
    import logging, glob
    # rename_name should match the conf in predictor.
    rename_name = 'tf_models'
    data_path = '../data'
    # model_dst is the path in norm_gard to send to.
    model_dst = ''
    with open('../model.des','r') as f:
      for line in f:
        line=line.strip()
        if line[0]=='#':
          continue
        key,value=line.split('=',1)
        if key=='destination':
          model_dst = value
          break
      else:
        logging.info('[F] Can\'t read destination folder,please check model.des.')
        raise Exception('Conf Error')
    # get dumped model folder.
    path = '../models'
    all_model_path = glob.glob(path + '/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/')
    not_null_model_path = [x for x in all_model_path if os.path.exists(os.path.join(x, 'export.index'))]
    if len(not_null_model_path) != 1:
        logging.info('[F] not exactly one model in ../models folder.')
        raise Exception('SYNC Error')
    model_path = not_null_model_path[0]
    logging.info('Found model: %s', model_path)
    logging.info('model path: %s', path)
    # do them
    rename_cmd = 'cd %s && rm %s ; cp -r %s %s && cd -' % (path, rename_name, model_path, rename_name)
    renamed_path = os.path.join(os.path.dirname(model_path[:-1]), rename_name)
    md5sum_cmd = 'cd %s && md5sum %s/* > %s.status && cd -' % (path, rename_name, rename_name)
    import time, datetime
    datetime_path = time.strftime('%Y%m%d%H%M', time.localtime())
    # do local
    local_model_path = os.path.join(path, datetime_path)
    local_model_path_2 = os.path.join(local_model_path, 'model')
    local_model_cmd = 'mkdir %s; mkdir %s' % (local_model_path, local_model_path_2)
    cp_model_cmd = 'cp -rf %s* %s' % (renamed_path, local_model_path_2)
    cp_data_cmd = 'cp -rf %s %s' % (data_path, local_model_path)
    prepare_norm_guard_data(datetime_path)

    # run cmd
    logging.info(rename_cmd)
    os.system(rename_cmd)
    logging.info(md5sum_cmd)
    os.system(md5sum_cmd)
    logging.info(local_model_cmd)
    os.system(local_model_cmd)
    logging.info(cp_model_cmd)
    os.system(cp_model_cmd)
    logging.info(cp_data_cmd)
    os.system(cp_data_cmd)
    # send local model to nfs
    scp_model_dir(local_model_path, datetime_path, model_dst, logging)


send_model()
# ==============
# from custom_cmds.cmds import BackupLogs
BackupLogs()


# def mvlogs():
#     import logging, time
#     datetime_path = time.strftime('%Y%m%d%H%M', time.localtime())
#     folder = '/tmp/app_home_unify/' + datetime_path
#     cmd = 'cp -r ../logs ' + folder
#
#     logging.info(cmd)
#     os.system(cmd)

# BackupLogs()
# BackupEnv()
