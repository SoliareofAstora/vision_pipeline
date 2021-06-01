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


def assign_jobs(free_queue_space, jobs, space_for_new_jobs):
    unassigned_job_keys = [key for key in jobs.keys() if 'host' not in jobs[key].keys()]
    assigned_job_keys = {}

    if space_for_new_jobs <= len(unassigned_job_keys):
        for host_device in free_queue_space.keys():
            selected_jobs = random.sample(unassigned_job_keys, free_queue_space[host_device])
            assigned_job_keys[host_device] = selected_jobs
            for key in selected_jobs:
                unassigned_job_keys.remove(key)
    else:
        # todo change it to assigning to shortest queue first.
        for job_key in unassigned_job_keys:
            n = random.randint(1, space_for_new_jobs)
            for host_device in free_queue_space.keys():
                n -= free_queue_space[host_device]
                if n <= 0:
                    space_for_new_jobs -= 1
                    free_queue_space[host_device] -= 1
                    if host_device not in assigned_job_keys.keys():
                        assigned_job_keys[host_device] = []
                    assigned_job_keys[host_device] = job_key
                    break

    return assigned_job_keys


def main(json_path, target_queue_size):
    json_path = pathlib.Path(json_path)
    jobs = json.load(open(json_path, 'r'))
    backup_jobs = copy.deepcopy(jobs)

    hosts = list(HOSTS_CONFIG.keys())
    hosts_devices = {}
    hosts_queue = {}
    hosts_completed = {}
    space_for_new_jobs = 0

    # ONLINE read current state on remote machines
    connected = []
    failed = []
    for host in hosts[:]:
        if not verify_connection(host, connected, failed):
            hosts.remove(host)
        else:
            prepare_folder_structure(host)
            # experiments_errors(host, jobs)
            if contain_key(host, 'device'):
                print(host)
                # print(execute_command(host, 'nvidia-smi'))
                hosts_devices[host] = HOSTS_CONFIG[host]['device']
                hosts_queue[host] = list_dirs(host, 'experiments/queue/')
                hosts_completed[host] = list_dirs(host, 'experiments/completed/')
                space_for_new_jobs += max(0, len(hosts_devices[host]) * target_queue_size - len(hosts_queue[host]))
            else:
                hosts.remove(host)

    # offline check available free queue space on each (host, device)
    free_queue_space = {(host, device): target_queue_size for host in hosts for device in hosts_devices[host]}
    for host in hosts:
        for completed in hosts_completed[host]:
            job_id = completed.split('/')[-1]
            if jobs[job_id]['status'] == 'queued' or jobs[job_id]['status'] == 'error':
                if jobs[job_id]['host'] != host:
                    print('Two machines were assigned the same task. Code is OK with it.')
                    jobs[job_id]['host'] = host
                jobs[job_id]['status'] = 'completed'

        for job_path in hosts_queue[host]:
            job_id = job_path.split('/')[-1]
            if 'host' not in jobs[job_id].keys():
                print('Some error eccured. Trying to repair')
                # todo fix it
            if jobs[job_id]['host'] != host:
                print('2 machines have queued the same task. To avoid waste of energy please implement something.')
                print(f'Or simply remove {job_id} from {host} or {jobs[job_id]["host"]}')
            job_device = jobs[job_id]['device']
            jobs[job_id]['status'] = 'queued'
            # todo check if device is still available in HOSTS_CONFIG
            free_queue_space[(host, job_device)] = max(0, free_queue_space[(host, job_device)] - 1)
    for key in list(free_queue_space.keys()):
        if free_queue_space[key] == 0:
            free_queue_space.pop(key)
    if len(free_queue_space) == 0:
        print('No empty job slots found. Exiting.')
        return

    # offline assign new jobs to each (host, device) and remove hosts without new jobs.
    assigned_job_keys = assign_jobs(free_queue_space, jobs, space_for_new_jobs)
    for host in hosts[:]:
        unassigned_host = True
        for device in hosts_devices[host][:]:
            if (host, device) not in assigned_job_keys.keys():
                hosts_devices[host].remove(device)
            else:
                unassigned_host = False
        if unassigned_host:
            hosts.remove(host)
            hosts_devices.pop(host)
    print(f'\n\nAssigned {space_for_new_jobs} new jobs across {len(hosts)} hosts')

    # offline prepare archives with new jobs
    for host in hosts:
        host_dir = pathlib.Path(data_path('localhost')) / 'tmp' / host
        host_dir.mkdir()
        for device in hosts_devices[host]:
            for key in assigned_job_keys[(host, device)]:
                job_dir = host_dir / key
                job_dir.mkdir()
                jobs[key]['device'] = device
                json.dump(jobs[key], open(job_dir / 'job_args.json', 'w'))
                jobs[key]['host'] = host
                jobs[key]['status'] = 'queued'
        compress('localhost', host_dir.parent / (host + '.tar.gz'), list_files('localhost', host_dir), host_dir)

    # ONLINE send and uncompress new jobs on remote machines
    rollbacks = []
    for host in hosts:
        print(f'Updating queue on {host}')
        attempts = 10
        succeed = upload_unpack(host, f'tmp/{host}.tar.gz', 'experiments/queue', 10)
        while not succeed:
            print(f'Failed. Attempts left: {attempts}')
            succeed = upload_unpack(host, f'tmp/{host}.tar.gz', 'experiments/queue', 10)
            attempts -= 1
            if attempts < 0:
                break
        if not succeed:
            rollbacks.append(host)
        else:
            print('OK!')

    # offline delete changes in jobs file for specific hosts
    for host in rollbacks:
        for device in hosts_devices[host]:
            for key in assigned_job_keys[(host, device)]:
                jobs[key].pop('host')
                jobs[key].pop('status')
                jobs[key]['device'] = device

    print('Saving backup')
    backup_path = json_path.parent / 'backup' / datetime.datetime.now().strftime(
        "distribute_jobs-%d_%m_%Y-%H_%M_%S.json")
    backup_path.parent.mkdir(exist_ok=True)
    json.dump(backup_jobs, open(backup_path, 'w'))
    json.dump(jobs, open(json_path, 'w'))

    # todo print commands in console to copy paste run work.py


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        json_path = 'experiments/jobs_list.json'
        target_queue_size = 10
        main(json_path, target_queue_size)
    else:
        if len(args) == 1:
            json_path = args[0]
            target_queue_size = 10
            main(json_path, target_queue_size)
        else:
            json_path = args[0]
            target_queue_size = int(args[1])
            main(json_path, target_queue_size)
