HOSTS_CONFIG = {
    'localhost': {'repo_path': '/home/soliareofastora/bakterie/', 'data_path': '/aml/bacteria/'},
    # 'neptune': {'device': ['cuda:0'], 'repo_path': '/home/soliareofastora/strains/', 'data_path': '/home/soliareofastora/data/'},
    'walter': {'device': ['cuda:0', 'cuda:1'], 'repo_path': '/home/pkucharski/strains/', 'data_path': '/storage/ssd_storage1/pkucharski/bacteria/'},
    'wmii': {'device': ['cuda:0', 'cuda:1'], 'proxy_host': 'walter', 'repo_path': '/home/pkucharski/strains/', 'data_path': '/media/data2/pkucharski/bacteria/'},
}


def repo_path(host):
    return HOSTS_CONFIG[host]['repo_path']


def data_path(host):
    return HOSTS_CONFIG[host]['data_path']


def contain_key(host, key):
    return key in HOSTS_CONFIG[host].keys()
