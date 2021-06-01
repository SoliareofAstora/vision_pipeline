import json
import pathlib
import random
import datetime
import copy
import sys

from remote_hosts.file_system import list_dirs, list_files, compress, upload_unpack, prepare_folder_structure
from remote_hosts.config import HOSTS_CONFIG, data_path, contain_key
from remote_hosts.verify_connection import verify_connection
from remote_hosts.execute_command import execute_command
from print_errors import experiments_errors


def main(json_path):
    json_path = pathlib.Path(json_path)
    jobs = json.load(open(json_path, 'r'))
    backup_jobs = copy.deepcopy(jobs)

    hosts = list(HOSTS_CONFIG.keys())
    hosts_completed = {}

    # ONLINE read current state on remote machines
    connected = []
    failed = []
    for host in hosts[:]:
        if not verify_connection(host, connected, failed):
            hosts.remove(host)
        else:
            prepare_folder_structure(host)
            experiments_errors(host, jobs)
            hosts_completed[host] = list_dirs(host, 'experiments/completed/')


    # todo finish this file

    for host in hosts[:]:
        for completed in hosts_completed[host][:]:
            job_id = completed.split('/')[-1]
            if jobs[job_id]['status'] == 'queued' or jobs[job_id]['status'] == 'error':
                if jobs[job_id]['host'] != host:
                    print('Two machines were assigned the same task. Code is OK with it.')
                    jobs[job_id]['host'] = host
                jobs[job_id]['status'] = 'completed'



    print('Saving backup')
    backup_path = json_path.parent / 'backup' / datetime.datetime.now().strftime(
        "distribute_jobs-%d_%m_%Y-%H_%M_%S.json")
    backup_path.parent.mkdir(exist_ok=True)
    json.dump(backup_jobs, open(backup_path, 'w'))
    json.dump(jobs, open(json_path, 'w'))