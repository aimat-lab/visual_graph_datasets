"""
This is a utility cli tool which allows to push large folders of data automatically in chunks. The problem
is that git only allows 2GB per push.
"""
import os
import sys
import time
import subprocess

import click


@click.command()
@click.argument('path')
@click.option('-l', '--limit', type=click.FLOAT, default=1.0,
              help='Size limit per commit in GB',)
@click.option('--remote', type=click.STRING, default='origin',
              help='The remote identifier to push to')
@click.option('--branch', type=click.STRING, default='master',
              help='The branch to be pushed')
@click.option('--batch-size', type=click.INT, default=1000,
              help='The number of files to add to git at same time')
def main(path: str,
         limit: int,
         remote: str,
         branch: str,
         batch_size: int):

    click.echo(f'pushing all the files from folder "{path}" to remote repository')

    start_time = time.time()
    fragment_counter = 0
    current_size = 0
    total_size = 0
    current_paths = []
    for root, folders, files in os.walk(path):

        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size / 1024**3
            current_size += file_size
            current_paths.append(file_path)

            # If the size of the current buffer exceeds the limit, we execute the routine to commit all
            # those files and push them.
            if current_size > limit:
                click.echo(f'adding {len(current_paths)} files to git...')
                # first of all we need to add all those files from the buffer to git
                current_index = 0
                while current_index < len(current_paths):
                    num_elements = min(len(current_paths) - current_index, batch_size)
                    current_batch = current_paths[current_index:current_index+num_elements]
                    add_command = ['git', 'add'] + current_batch
                    proc: subprocess.CompletedProcess = subprocess.run(
                        add_command,
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if proc.returncode == 0:
                        click.echo(f' * {current_index}/{len(current_paths)} done')
                        current_index += num_elements
                    else:
                        click.echo('error:')
                        click.echo(proc.stderr.decode())

                # Then we need to form the commit
                commit_command = ['git', 'commit', '-am', f'"large push fragment {fragment_counter}"']
                subprocess.run(commit_command, shell=False, stdout=sys.stdout)

                # Ultimately we need to push
                click.echo('pushing...')
                push_command = ['git', 'push', remote, branch]
                subprocess.run(push_command, shell=False, stdout=sys.stdout)

                click.echo(f'pushed fragment {fragment_counter}'
                           f' - elapsed time: {time.time() - start_time:.1f} seconds')
                fragment_counter += 1
                total_size += current_size
                current_size = 0
                current_paths = []

    click.secho(f'finished pushing {total_size} GB in {time.time() - start_time:.1f} seconds', fg='green')


if __name__ == '__main__':
    main()
