import argparse
import json
import pathlib
import random
import shutil
import time
import traceback
from datetime import datetime
from multiprocessing import cpu_count

from lock import lock, unlock
from training import train_cnn_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='Iterate over queue and do the jobs')
    parser.add_argument('-data_path', type=str, required=True)
    parser.add_argument('-device', type=str, required=True)
    parser.add_argument('-core_limit', help='limit cores used across pipeline', type=int, default=0, required=False)
    return parser.parse_args()


def move(source, destination, exist_ok=True):
    lock(destination.parent)
    lock(source.parent)

    shutil.move(str(source), str(destination))

    unlock(destination.parent)
    unlock(source.parent)


def get_job_function(job_type):
    if job_type == 'train_cnn':
        return train_cnn_classifier.train
    raise KeyError


def load_queue(queue_path, device):
    queued_experiments = [{'key': x.name, 'args': json.load(open(x / 'job_args.json', 'r'))} for x in queue_path.iterdir() if x.is_dir()]
    device_experiments = [e for e in queued_experiments if e['args']['device'] == device]
    if len(device_experiments) > 0:
        return device_experiments
    else:
        return queued_experiments


def main(data_path, device, core_limit=0):
    data_path = pathlib.Path(data_path)
    while True:
        queue_path = data_path / 'experiments' / 'queue'
        lock(queue_path)
        queue = load_queue(queue_path, device)
        if len(queue) == 0:
            print("Queue empty. Waiting for next jobs")
            timeout = time.time() + 24 * 2 * 60 * 60
            while len(queue) == 0:
                unlock(queue_path)
                if time.time() > timeout:
                    raise TimeoutError('Queue was empty for 2 days. Terminating.')
                time.sleep(60)
                lock(queue_path)
                queue = load_queue(queue_path, device)

        job = random.choice(queue)
        job_id = job['key']
        job_queue_path = data_path / 'experiments' / 'queue' / job_id
        job_work_path = data_path / 'experiments' / 'in_progress' / job_id
        job_done_path = data_path / 'experiments' / 'completed' / job_id

        experiment_args = job['args']['experiment_args']
        experiment_args['work_dir'] = job_work_path
        experiment_args['data_path'] = data_path
        experiment_args['device'] = device
        experiment_args['job_id'] = job_id
        experiment_args['core_limit'] = cpu_count() if core_limit == 0 else core_limit
        job_function = get_job_function(experiment_args['job_type'])

        lock(job_work_path.parent)
        job_work_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job_queue_path), str(job_work_path))
        unlock(queue_path)
        unlock(job_work_path.parent)

        print(f"Working on {job_id}. Experiments left in queue: {str(len(queue))}")
        try:
            job_function(experiment_args)
            move(job_work_path, job_done_path)

        except (SystemExit, KeyboardInterrupt):
            # terminate entire work process
            move(job_work_path, job_queue_path)
            return 0

        except TimeoutError:
            # move experiment back to queue
            exception_time = datetime.now().strftime("%d/%m/%Y at %H:%M")
            print(f'{exception_time} job_id: {job_id} TIMEOUT')
            move(job_work_path, job_queue_path)
            # todo count timeout exceptions and throw error if count exceeds, for example, 3
            # todo and implement it overall like all other exceptions. Save traceback and stuff.
            continue

        except Exception:
            # save error info and move experiment to error
            exception_time = datetime.now().strftime("%d/%m/%Y at %H:%M")
            print(f'{exception_time} job_id: {job_id}, ARGS: {experiment_args}')
            job_error_path = data_path / 'experiments' / 'error' / job_id
            move(job_work_path, job_error_path)
            trace = traceback.format_exc()
            print(trace)
            with open(job_error_path / 'error_info.txt', 'a') as f:
                f.write(exception_time + '\n')
                f.write(trace)
                f.write('\n' * 5)


if __name__ == '__main__':
    args = parse_args()
    main(args.data_path, args.device, args.core_limit)
