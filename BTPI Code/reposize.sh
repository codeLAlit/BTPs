#!/bin/bash
if [ $# -ne 3 ]
then
  echo "Usage: gitrepo-info <username> <owner> <repo>"
  exit 65
fi
curl -u "$1" http://github.com/api/v2/json/repos/show/$2/$3
