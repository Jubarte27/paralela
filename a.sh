## possibly useful for finding processes that are orphaned and have been adopted by init (pid 1)
ps -eo pid,ppid,cmd | awk '$2==1'


# Maybe make the parameters named, but not today
IFS=, read -r -a headers < "$CSV_FILE"
for i in "${!headers[@]}"; do true; done
param_name=$(echo "${headers[$i]}" | tr -d '\r')
"--$param_name"


# Hope hype doesnt need this
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope