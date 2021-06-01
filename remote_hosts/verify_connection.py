from remote_hosts.config import HOSTS_CONFIG, contain_key
from remote_hosts.execute_command import execute_command


def verify_connection(host, connected=None, failed=None):
    if connected is None:
        connected = []
    if host in connected:
        return True
    if failed is None:
        failed = []
    if host in failed:
        return False
    print(f'Trying to connect to {host}')
    if contain_key(host, 'proxy_host'):
        proxy_host = HOSTS_CONFIG[host]['proxy_host']
        print(f"Verifying proxy_host {proxy_host} while trying to connect to {host}")
        if not verify_connection(proxy_host, connected, failed):
            print(f"Failed to connect to proxy_host {proxy_host}")
            failed.append(host)
            return False
        print(f"Connected to proxy_host {host}")
    try:
        verify = execute_command(host, 'uname -a', 5)
    except (TimeoutError, RuntimeError):
        print(f"Failed to connect to {host}")
        failed.append(host)
        return False

    print(f"Connected to {host}: {verify[:-1]}")
    connected.append(host)
    return True


