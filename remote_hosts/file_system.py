from remote_hosts.config import HOSTS_CONFIG, repo_path, data_path, contain_key
from remote_hosts.execute_command import execute_command


# todo refactor this function. I bet there is more elegant way.
def copy_file(source_host, destination_host, local_file_path, timeout=3600):
    if source_host == 'localhost' and destination_host == 'localhost':
        return

    source_path = data_path(source_host) + local_file_path
    destination_path = data_path(destination_host) + local_file_path

    if not contain_key(source_host, 'proxy_host') and not contain_key(destination_host, 'proxy_host'):
        command = f"scp {source_host}:{source_path} {destination_host}:{destination_path}".replace('localhost:', '')
        execute_command('localhost', command, timeout)

    else:
        if contain_key(source_host, 'proxy_host') and 'proxy_host' in contain_key(destination_host, 'proxy_host'):
            copy_file(source_host, HOSTS_CONFIG[source_host]['proxy_host'], local_file_path, timeout)
            copy_file(HOSTS_CONFIG[source_host]['proxy_host'], destination_host, local_file_path, timeout)

        elif contain_key(source_host, 'proxy_host'):
            proxy_host = HOSTS_CONFIG[source_host]['proxy_host']
            proxy_path = data_path(proxy_host) + local_file_path

            command = f"scp {source_host}:{source_path} {proxy_path}"
            execute_command(proxy_host, command, timeout)

            command = f"scp {proxy_host}:{proxy_path} {destination_host}:{destination_path}".replace('localhost:', '')
            execute_command('localhost', command, timeout)

        elif contain_key(destination_host, 'proxy_host'):
            proxy_host = HOSTS_CONFIG[destination_host]['proxy_host']
            proxy_path = data_path(proxy_host) + local_file_path

            command = f"scp {source_host}:{source_path} {proxy_host}:{proxy_path}".replace('localhost:', '')
            execute_command('localhost', command, timeout)

            command = f"scp {proxy_path} {destination_host}:{destination_path}"
            execute_command(proxy_host, command, timeout)

    if get_file_size(source_host, local_file_path) != get_file_size(destination_host, local_file_path):
        raise RuntimeError("Copied file size doesn't match original file size")


def upload(destination_host, local_file_path, timeout=3600):
    copy_file('localhost', destination_host, local_file_path, timeout)


def download(source_host, local_file_path, timeout=3600):
    copy_file(source_host, 'localhost', local_file_path, timeout)


def list_all(host, path):
    path = str(path)
    if not path.startswith('/'):
        path = data_path(host) + path
    command = f"find {path} -maxdepth 1"
    return execute_command(host, command).split('\n')[1:-1]


def list_dirs(host, path, recursive=False):
    path = str(path)
    if not path.startswith('/'):
        path = data_path(host) + path
    command = f"find {path}{'' if recursive else ' -maxdepth 1'} -type d"
    return execute_command(host, command).split('\n')[1:-1]


def list_files(host, path, pattern=None, recursive=True):
    path = str(path)
    if not path.startswith('/'):
        path = data_path(host) + path
    command = f"find {path} -type f"
    if not recursive:
        command += ' -maxdepth 1'
    if pattern is not None:
        command += f' -name {pattern}'
    return execute_command(host, command).split('\n')[:-1]


def compress(host, targz_path, absolute_file_paths, relative_to=None, timeout=3600):
    targz_path = str(targz_path)
    if not targz_path.startswith('/'):
        targz_path = data_path(host) + targz_path
    if not targz_path.endswith('.tar.gz'):
        targz_path = targz_path + '.tar.gz'

    if relative_to is None:
        relative_to = data_path(host)[:-1]
    else:
        relative_to = str(relative_to)
        if not relative_to.startswith('/'):
            relative_to = data_path(host) + relative_to

    files_str = ' '.join([path.replace(relative_to + '/', '') for path in absolute_file_paths])
    command = f"tar -C {relative_to} -czf {targz_path} {files_str}"
    execute_command(host, command, timeout)


def uncompress(host, targz_path, relative_to_local=None, timeout=3600):
    targz_path = str(targz_path)
    if not targz_path.startswith('/'):
        targz_path = data_path(host) + targz_path
    if relative_to_local is None:
        relative_to_local = data_path(host)[:-1]
    else:
        relative_to_local = data_path(host) + relative_to_local + '/'
    command = f"tar -C {relative_to_local} -xzf {targz_path}"
    execute_command(host, command, timeout)


def lock(host, local_path, timeout=10):
    command = f"python3 {repo_path(host)}lock.py lock {data_path(host) + local_path} {timeout}"
    execute_command(host, command, timeout + 5)


def unlock(host, local_path):
    command = f"python3 {repo_path(host)}lock.py unlock {data_path(host) + local_path}"
    execute_command(host, command)


def get_file_size(host, local_file_path):
    return int(execute_command(host, f"stat -c %s {data_path(host) + local_file_path}"))


def mkdir(host, path):
    path = str(path)
    if not path.startswith('/'):
        path = data_path(host) + path
    execute_command(host, f'mkdir -p {path}')


def rm(host, path, recursive=False, timeout=10):
    path = str(path)
    if not path.startswith('/'):
        path = data_path(host) + path
    execute_command(host, f'rm{" -r " if recursive else " "}{path}', timeout)


def upload_unpack(host, local_file_path, local_uncompress_root, timeout=600):
    try:
        upload(host, local_file_path, timeout)
    except (RuntimeError, TimeoutError)as e:
        print(e)
        print(f'problem with sending file to {host}, skipping')
        return False

    try:
        lock(host, local_uncompress_root)
    except (RuntimeError, TimeoutError)as e:
        print(e)
        print(f'Unable to lock {local_uncompress_root} on {host}')
        try:
            unlock(host, local_uncompress_root)
        except (RuntimeError, TimeoutError)as e:
            print(e)
            print(f'MANUAL REPAIR MIGHT BE REQUIRED ON {host}')
        return False

    try:
        uncompress(host, local_file_path, local_uncompress_root, timeout)
    except (RuntimeError, TimeoutError) as e:
        print(e)
        print(f'FAILED TO UNCOMPRESS FILES! MANUAL REPAIR MIGHT BE REQUIRED ON {host}')
        try:
            unlock(host, local_uncompress_root)
        except (RuntimeError, TimeoutError)as e:
            print(e)
            print(f'MANUAL REPAIR IS REQUIRED ON {host}')
        return False

    try:
        unlock(host, local_uncompress_root)
    except (RuntimeError, TimeoutError)as e:
        print(e)
        print(f'ERROR WHILE UNLOCKING {local_uncompress_root} ON {host}')
        return False
    return True


def compress_download_unpack(host, remote_files_absolute_path, local_uncompress_root):
    compress(host, f'tmp/{host}.tar.gz', remote_files_absolute_path)
    download(host, f'tmp/{host}.tar.gz')
    mkdir('localhost', data_path('localhost') + local_uncompress_root)
    uncompress('localhost', f'tmp/{host}.tar.gz', local_uncompress_root)


def prepare_folder_structure(host):
    root = data_path(host)
    paths = ['datasets', 'experiments', 'tmp']
    dirs = list_dirs(host, root)
    for path in paths:
        if root + path not in dirs:
            mkdir(host, root + path)

    tmp_contents = list_all(host, root + 'tmp')
    if len(tmp_contents) > 0:
        rm(host, ' '.join(tmp_contents), recursive=True)

    paths = ['experiments/completed', 'experiments/error', 'experiments/in_progress', 'experiments/queue']
    dirs = list_dirs(host, root + 'experiments')
    for path in paths:
        if root + path not in dirs:
            mkdir(host, root + path)
