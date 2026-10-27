printf ready > ready
while [ ! -f release ]; do sleep 0.3; done
printf finished\n
