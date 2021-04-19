+ cd /lustre/haven/proj/UTK0150/atown/assignment4
+ echo 'changed into working directory'
++ head -n 1 /var/spool/torque/aux//4747416.apollo-acf
+ export MASTER_ADDR=acf-sc100
+ MASTER_ADDR=acf-sc100
+ export MASTER_PORT=12345
+ MASTER_PORT=12345
++ which python
+ mpirun -n 1 /lustre/haven/proj/UTK0150/envs/pytorch181/bin/python assignment4.py
+ echo 'done running python file.'
