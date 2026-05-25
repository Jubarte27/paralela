## possibly useful for finding processes that are orphaned and have been adopted by init (pid 1)
ps -eo pid,ppid,cmd | awk '$2==1'