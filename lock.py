#!/usr/bin/env python
import os
import pathlib
import sys
import time


def lock(path, timeout=60):
    timeout = time.time() + timeout
    if os.path.exists(path / 'lockfile'):
        print(f"Waiting for {str(path)} unlock")
    while os.path.exists(path / 'lockfile'):
        if time.time() > timeout:
            raise TimeoutError(f'Unable to lock {path}')
        time.sleep(5)
    open(path / 'lockfile', 'w')


def unlock(path):
    if os.path.exists(path / 'lockfile'):
        os.remove(path / 'lockfile')


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        command = args[0]
        path = pathlib.Path(args[1])
        if command == 'lock':
            if len(args) == 3:
                timeout = int(args[2])
                lock(path, timeout)
            else:
                lock(path)
            return
        elif command == 'unlock':
            unlock(path)
            return
    raise RuntimeError('Usage: python lock.py [lock/unlock] absolute_path')


if __name__ == '__main__':
    main()
