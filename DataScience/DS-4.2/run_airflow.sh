#!/bin/bash

airflow webserver -p 8080 &
xterm -e "airflow scheduler" &