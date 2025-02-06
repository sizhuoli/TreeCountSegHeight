#!/bin/bash

USER_UID=$(id -u)
USER_GID=$(id -g)

chmod -R 777 /app/predictions/ /app/saved_models/

chown -R $USER_UID:$USER_GID /app/predictions/ /app/saved_models/

exec "$@"
