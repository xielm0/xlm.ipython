#!/usr/bin/expect

username=admin
pass=NDdmZDQwNDBjZjYyOWYxNmIzZmUwZTEy

ip1=172.18.161.13
ip2=172.18.161.27

host=$ip1
echo $host

#auto login
spawn ssh admin@$host
set timeout 3 

expect "*password:"
send "$PASS\n"

interact