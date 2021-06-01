from remote_hosts.config import HOSTS_CONFIG
from remote_hosts.file_system import list_files, compress_download_unpack, rm
from remote_hosts.verify_connection import verify_connection


def experiments_errors(host, jobs=None):
    failed_experiments = list_files(host, 'experiments/error/', pattern='error_info.txt')
    if len(failed_experiments)>0:
        compress_download_unpack(host, failed_experiments, f'tmp/errors_info/{host}')
        error_info_paths = list_files('localhost', f'tmp/errors_info/{host}/')
        for error_path in sorted(error_info_paths):
            print(error_path.split('/')[-2])  # host
            job_id = error_path.split('/')[-2]
            print(job_id)
            if jobs is not None:
                jobs[job_id]['state'] = 'error'
            with open(error_path, 'r') as f:
                print(f.read())
            print('\n\n')
        rm('localhost', 'tmp/errors_info/', recursive=True)


def main():
    connected = []
    failed = []
    hosts = list(HOSTS_CONFIG.keys())
    for host in hosts[:]:
        if verify_connection(host, connected, failed):
            experiments_errors(host)


if __name__ == '__main__':
    main()
