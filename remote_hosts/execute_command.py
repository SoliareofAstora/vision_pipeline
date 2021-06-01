import subprocess

from remote_hosts.config import HOSTS_CONFIG, contain_key


def execute_command(host, command, timeout=60):
    if type(command) == str:
        command = str.split(command, ' ')
    if host != 'localhost':
        if not contain_key(host, 'proxy_host'):
            command = ['ssh', host] + command
        else:
            proxy_host = HOSTS_CONFIG[host]['proxy_host']
            if contain_key(proxy_host, 'proxy_host'):
                raise NotImplementedError(f"Dont use multiple proxy_host steps. Change {host} or {proxy_host}")
            command = ['ssh', proxy_host, 'ssh', host] + command

    try:
        completed_process = subprocess.run(command, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"command {' '.join(command)} timeout")

    if completed_process.stderr != b'':
        error_info = completed_process.stderr.decode()
        raise RuntimeError(f"during execution: {' '.join(command)} exception occurred\n{error_info}")
    else:
        return completed_process.stdout.decode('utf-8')