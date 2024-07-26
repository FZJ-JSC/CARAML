from argparse import ArgumentParser
import ipaddress
import runpy
import socket
import torch
from torch.distributed.elastic.agent.server import api as sapi


def parse_host():
    parser = ArgumentParser()
    parser.add_argument('--rdzv_endpoint')
    endpoint = parser.parse_known_args()[0].rdzv_endpoint
    host = (
        endpoint.split(':', 1)[0]
        if endpoint
        else None
    )
    return host


def fix_torch_run(host):
    _orig_get_fq_hostname = sapi._get_fq_hostname

    if host:
        try:
            ipaddress.ip_address(host)
            is_ip = True
        except ValueError:
            is_ip = False

        if is_ip:
            def new_get_fq_hostname():
                return socket.gethostbyaddr(host)[0]
        else:
            def new_get_fq_hostname():
                return socket.getfqdn(host)
    else:
        new_get_fq_hostname = _orig_get_fq_hostname

    sapi._get_fq_hostname = new_get_fq_hostname


def main():
    host = parse_host()
    fix_torch_run(host)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()