import moxing as mox
import sys

temp_dir = '' # sys.argv[1]
mox.file.copy_parallel(f'/home/ma-user/work/projects/Fot/code/forest-of-thought', f's3://bucket-4031/bizhenni/projects/chain_of_thought/forest-of-thought')
